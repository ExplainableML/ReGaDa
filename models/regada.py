import torch
import torch.nn as nn
import torch.nn.functional as F

from models.text_embedder import AdverbActionTextEmbedder
from models.video_embedder import VideoEmbedder


class ReGaDaModel(nn.Module):
    def __init__(
        self,
        dset,
        emb_dim,
        video_embedder,
        text_embedder,
        word_embedding,
        loss_triplet_action,
        loss_triplet_adverb,
        loss_l2,
        loss_triplet_margin,
        eval_no_act_gt,
        **args
    ):
        super().__init__()
        self.emb_dim = emb_dim
        self.dset = dset
        self.loss_triplet_action = loss_triplet_action
        self.loss_triplet_adverb = loss_triplet_adverb
        self.loss_l2 = loss_l2

        self.loss_triplet_margin = loss_triplet_margin
        self.eval_no_act_gt = eval_no_act_gt

        assert loss_triplet_action > 0 or loss_triplet_adverb > 0 or loss_l2 > 0
        self.dset = dset

        self.text_embedder = AdverbActionTextEmbedder(
            dset, out_dim=emb_dim, word_embedding=word_embedding, **text_embedder
        )
        self.video_embedder = VideoEmbedder(
            input_dim=dset.feature_dim,
            out_dim=emb_dim,
            word_dim=self.text_embedder.word_dim,
            **video_embedder
        )

        self.pairidx2idx_tensor = torch.tensor(
            self.dset.pairidx2idx_array, dtype=torch.long
        ).cuda()

    def scoring_func(self, vidfeat, action_adv_emb, normalize=True):
        # cosine similarity video embedding <--> action-adverb embeddings
        if normalize:
            vidfeat = F.normalize(vidfeat, dim=1)
            action_adv_emb = F.normalize(action_adv_emb, dim=-1)
        if action_adv_emb.dim() > 1:
            action_adv_emb = action_adv_emb.T
        cosine_sim = torch.matmul(vidfeat, action_adv_emb)

        return cosine_sim

    def compute_loss(
        self,
        video_embedding,
        positive_embeddings,
        negative_act_embeddings,
        negative_adv_embeddings,
    ):
        losses = []

        loss_triplet_act = torch.tensor(0)
        if self.loss_triplet_action > 0:
            loss_triplet_act = F.triplet_margin_loss(
                video_embedding,
                positive_embeddings,
                negative_act_embeddings,
                margin=self.loss_triplet_margin,
            )
            losses.append(loss_triplet_act * self.loss_triplet_action)

        loss_triplet_adv = torch.tensor(0)
        if self.loss_triplet_adverb > 0:
            loss_triplet_adv = F.triplet_margin_loss(
                video_embedding,
                positive_embeddings,
                negative_adv_embeddings,
                margin=self.loss_triplet_margin,
            )
            losses.append(loss_triplet_adv * self.loss_triplet_adverb)

        loss_l2 = torch.tensor(0)
        if self.loss_l2 > 0:
            loss_l2 = F.mse_loss(video_embedding, positive_embeddings)
            losses.append(loss_l2 * self.loss_l2)

        if len(losses) < 2:
            losses.append(torch.tensor(0))

        total_loss = sum(losses)
        loss_dict = {
            "loss_total": total_loss.detach().cpu(),
            "loss_triplet_action": loss_triplet_act.detach().cpu(),
            "loss_triplet_adverb": loss_triplet_adv.detach().cpu(),
            "loss_l2": loss_l2.detach().cpu(),
        }
        return total_loss, loss_dict

    def train_forward(self, x):
        features, actions = x[0], x[2]
        pad = x[4]

        (
            _,
            positive_embeddings,
            negative_act_embeddings,
            negative_adv_embeddings,
        ) = self.text_embedder(x)

        gt_action = self.text_embedder.text_embedder.get_action_embedding(actions)
        video_embedding = self.video_embedder(
            features,
            pad,
            action_embedding=gt_action,
        )

        loss, loss_dict = self.compute_loss(
            video_embedding,
            positive_embeddings,
            negative_act_embeddings,
            negative_adv_embeddings,
        )
        return loss, None, loss_dict

    def val_forward(self, x):
        features, actions = x[0], x[2]
        pad = x[4]

        # embed all possible action-adverb pairs
        text_embeddings = self.text_embedder(x)
        (
            action_adverb_embeddings,
            positive_embeddings,
            negative_act_embeddings,
            negative_adv_embeddings,
        ) = text_embeddings

        gt_action = self.text_embedder.text_embedder.get_action_embedding(actions)
        video_embedding = self.video_embedder(features, pad, action_embedding=gt_action)

        loss, loss_dict = self.compute_loss(
            video_embedding,
            positive_embeddings,
            negative_act_embeddings,
            negative_adv_embeddings,
        )

        if not self.eval_no_act_gt:
            scores = {}
            for i, (adverb, action) in enumerate(self.dset.pairs):
                score = self.scoring_func(video_embedding, action_adverb_embeddings[i])
                scores[(adverb, action)] = score
        else:
            scores = self.get_predictions(features, pad, text_embeddings).detach().cpu()

        return loss, scores, loss_dict

    def forward(self, x, **args):
        loss_dict = None
        if self.training:
            loss, pred, loss_dict = self.train_forward(x)
        else:
            with torch.no_grad():
                loss, pred, loss_dict = self.val_forward(x)
        return loss, pred, loss_dict

    def get_predictions(self, video_features, pad, text_embeddings):
        batch_size = video_features.shape[0]
        pair_scores = torch.zeros(
            (batch_size, len(self.dset.pairs)), device=video_features.device
        )
        action_adverb_embeddings = text_embeddings[0]

        gt_actions = (
            self.text_embedder.text_embedder.get_action_embedding(
                torch.arange(len(self.dset.idx2action), device=video_features.device)
            )
            .unsqueeze(1)
            .expand(-1, batch_size, -1)
        )

        for verb_idx, verb_str in self.dset.idx2action.items():
            video_embedding = self.video_embedder(
                video_features,
                pad,
                action_embedding=gt_actions[verb_idx],
            )
            score = self.scoring_func(
                video_embedding,
                action_adverb_embeddings[self.pairidx2idx_tensor[:, verb_idx]],
            )

            pair_scores[:, self.pairidx2idx_tensor[:, verb_idx]] = score

        return pair_scores
