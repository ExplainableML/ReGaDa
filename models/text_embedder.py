import torch
import torch.nn as nn

from models.word_embedding import load_word_embeddings


class AdverbActionTextEmbedder(nn.Module):
    def __init__(self, dset, out_dim, word_embedding, **args):
        super().__init__()
        self.dset = dset
        self.text_embedder = ResidualGateText(dset, out_dim, word_embedding, **args)
    
        self.word_dim = self.text_embedder.action_embedder.weight.shape[-1]

    def forward(self, inputs):
        x = self.text_embedder(inputs)
        return x


class ResidualGateText(nn.Module):
    def __init__(
        self,
        dset,
        out_dim,
        word_embedding,
        dropout,
        composition,
        tokens,
        main_modal,
        layers_gated,
        layers_residual,
        **args,
    ):
        super().__init__()
        self.dset = dset
        self.composition_func = composition
        self.main_modal = main_modal

        (
            self.action_embedder,
            self.adverb_embedder,
            self.pair_embedder,
            word_dim,
        ) = self.init_embeddings(word_embedding)

        self.dset = dset
        self.tokens = tokens

        num_tokens = 0
        num_tokens += 1 if "act" in self.tokens else 0
        num_tokens += 1 if "adv" in self.tokens else 0
        num_tokens += 1 if "comp" in self.tokens else 0
        # map word embeddings to output dim
        word_proj_adv, word_proj_act, word_proj_comp = (
            nn.Identity(),
            nn.Identity(),
            nn.Identity(),
        )
        if word_dim != out_dim:
            # projecting actions / adverbs / compositions separately
            if "act" in self.tokens:
                word_proj_act = nn.Linear(word_dim, out_dim)
            else:
                word_proj_act = nn.Identity()
            if "adv" in self.tokens:
                word_proj_adv = nn.Linear(word_dim, out_dim)
            else:
                word_proj_adv = nn.Identity()
            if "comp" in self.tokens:
                word_proj_comp = nn.Linear(word_dim, out_dim)
            else:
                word_proj_comp = nn.Identity()
        self.word_proj = nn.ModuleDict(
            {"act": word_proj_act, "adv": word_proj_adv, "comp": word_proj_comp}
        )

        residual_dim_in = out_dim
        residual_dim_out = out_dim
        residual_dim = residual_dim_in * (num_tokens - 1)
        self.text_projector = ResidualGate(
            residual_dim_in,
            residual_dim_out,
            residual_dim,
            dropout,
            layers_gated,
            layers_residual,
            init_weight=[1.0, 1.0],
        )

        ## precompute validation pairs
        adverbs, actions = zip(*self.dset.pairs)
        self.val_pairs = torch.LongTensor(range(len(self.dset.pairs))).cuda()
        self.val_adverbs = torch.LongTensor(
            [dset.adverb2idx[adv.strip()] for adv in adverbs]
        ).cuda()
        self.adverbs = torch.LongTensor(
            [dset.adverb2idx[adv.strip()] for adv in self.dset.adverbs]
        ).cuda()
        self.val_actions = torch.LongTensor(
            [dset.action2idx[act.strip()] for act in actions]
        ).cuda()

        self.pairidx2idx = {}
        for idx, pair_orig in enumerate(dset.pairs):
            pair = (
                dset.adverb2idx[pair_orig[0].strip()],
                dset.action2idx[pair_orig[1].strip()],
            )
            self.pairidx2idx[pair] = idx
        self.pairidx2idx_tensor = (
            torch.zeros(
                (len(self.dset.adverbs), len(dset.actions)), dtype=torch.long
            ).cuda()
            - 1
        )
        for idx, pair_orig in enumerate(dset.pairs):
            self.pairidx2idx_tensor[
                dset.adverb2idx[pair_orig[0].strip()],
                dset.action2idx[pair_orig[1].strip()],
            ] = idx

    def init_embeddings(self, word_embedding):
        vocab = self.dset.actions + self.dset.adverbs
        if self.composition_func == "emb":
            vocab += [f"{adv} {act}" for adv, act in self.dset.pairs]

        pretrained_weight = load_word_embeddings(
            word_embedding, vocab, self.dset.feature_dir
        )
        word_embedding_dim = pretrained_weight.shape[1]

        action_embedder = nn.Embedding(len(self.dset.actions), word_embedding_dim)
        adverb_embedder = nn.Embedding(len(self.dset.adverbs), word_embedding_dim)

        action_embedder.weight.data.copy_(pretrained_weight[: len(self.dset.actions)])
        adverb_embedder.weight.data.copy_(
            pretrained_weight[
                len(self.dset.actions) : len(self.dset.actions) + len(self.dset.adverbs)
            ]
        )
        for param in action_embedder.parameters():
            param.requires_grad = False
        for param in adverb_embedder.parameters():
            param.requires_grad = False

        if self.composition_func == "emb":
            pair_embedder = nn.Embedding(len(self.dset.pairs), word_embedding_dim)
            pair_embedder.weight.data.copy_(
                pretrained_weight[len(self.dset.actions) + len(self.dset.adverbs) :]
            )
            for param in pair_embedder.parameters():
                param.requires_grad = False
        else:
            pair_embedder = None

        return action_embedder, adverb_embedder, pair_embedder, word_embedding_dim

    def get_action_embedding(self, action_idx):
        act_embedding = self.action_embedder(action_idx)
        return act_embedding

    def _composition_func(self, action_embeddings, adverb_embeddings):
        if self.composition_func == "sum":
            pair_embeddings = action_embeddings + adverb_embeddings
        elif self.composition_func == "emb":
            pair_embeddings = self.pair_embedder(self.val_pairs)
        else:
            raise NotImplementedError

        return pair_embeddings

    def forward(self, x):
        adverbs, actions, pairs = x[1], x[2], x[3]
        neg_adverbs, neg_actions = x[5], x[6]

        # compose action-adverbs-pairs like a batch
        action_embeddings = self.action_embedder(self.val_actions)
        adverb_embeddings = self.adverb_embedder(self.val_adverbs)

        pair_embeddings = None
        if "comp" in self.tokens:
            # create composition embedding and add to input
            pair_embeddings = self._composition_func(
                action_embeddings, adverb_embeddings
            )

        remain_token = self.tokens.replace(self.main_modal, "")
        choice_dict = {
            "act": action_embeddings,
            "adv": adverb_embeddings,
            "comp": pair_embeddings,
        }
        residual_input = []
        residual_input.append(
            self.word_proj[self.main_modal](choice_dict[self.main_modal])
        )
        if "act" in remain_token:
            residual_input.append(self.word_proj["act"](action_embeddings))
        if "adv" in remain_token:
            residual_input.append(self.word_proj["adv"](adverb_embeddings))
        if "comp" in remain_token:
            residual_input.append(self.word_proj["comp"](pair_embeddings))

        output_embeddings = self.text_projector(residual_input)

        positive_embeddings = output_embeddings[pairs]
        negative_act_embeddings = output_embeddings[
            self.pairidx2idx_tensor[adverbs, neg_actions]
        ]
        negative_adv_embeddings = output_embeddings[
            self.pairidx2idx_tensor[neg_adverbs, actions]
        ]

        return (
            output_embeddings,
            positive_embeddings,
            negative_act_embeddings,
            negative_adv_embeddings,
        )


class ConCatModule(nn.Module):
    def __init__(self):
        super(ConCatModule, self).__init__()

    def forward(self, x):
        x = torch.cat(x, dim=1)
        return x


class ResidualGate(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        residual_dim,
        dropout,
        layers_gated,
        layers_residual,
        init_weight=[1.0, 1.0],
    ):
        super().__init__()

        self.weight_gate = nn.Parameter(torch.tensor(init_weight))
        dim_inner = dim_in + residual_dim

        gated_feature_composer = []
        gated_feature_composer.append(ConCatModule())
        gated_feature_composer.append(nn.BatchNorm1d(dim_inner))
        for i in range(1, layers_gated):
            gated_feature_composer.append(nn.Linear(dim_inner, dim_inner))
            gated_feature_composer.append(nn.Dropout(dropout))
            gated_feature_composer.append(nn.LeakyReLU(0.2))

        gated_feature_composer.append(nn.Linear(dim_inner, dim_out))

        self.gated_feature_composer = nn.Sequential(*gated_feature_composer)

        res_info_composer = []
        res_info_composer.append(ConCatModule())
        res_info_composer.append(nn.BatchNorm1d(dim_inner))
        for i in range(1, layers_residual):
            res_info_composer.append(nn.Linear(dim_inner, dim_inner))
            res_info_composer.append(nn.Dropout(dropout))
            res_info_composer.append(nn.LeakyReLU(0.2))
        res_info_composer.append(nn.Linear(dim_inner, dim_out))

        self.res_info_composer = nn.Sequential(*res_info_composer)

    def forward(self, modalities):
        assert len(modalities) > 1

        modal_main = modalities[0]
        modal_auxilary = torch.cat(modalities[1:], dim=1)

        # residual gating
        f1 = self.gated_feature_composer((modal_main, modal_auxilary))

        f2 = self.res_info_composer((modal_main, modal_auxilary))

        feat_main = (
            torch.sigmoid(f1) * modal_main * self.weight_gate[0]
            + f2 * self.weight_gate[1]
        )

        return feat_main
