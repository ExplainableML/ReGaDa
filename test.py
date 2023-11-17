import logging
import os
from collections import defaultdict

import hydra
import numpy as np
import torch
from hydra.utils import get_original_cwd, to_absolute_path
from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import average_precision_score
from tqdm.auto import tqdm

from dataset import setup_s3d_data
from models import get_model
from models.evaluator import Evaluator
from utils import AverageMeter, get_git_revision_hash, metric_string_mapping

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver(
    "basename", lambda x: os.path.basename(os.path.abspath(x)), replace=True
)


def test(args, model, data_loader, test_set, evaluator):
    model.eval()
    test_metrics = defaultdict(float)
    loss_logging_dict = defaultdict(AverageMeter)

    y_true_adverb = np.zeros((len(test_set), len(test_set.adverbs)))
    y_score = np.zeros((len(test_set), len(test_set.adverbs)))
    y_score_antonym = np.zeros((len(test_set), len(test_set.adverbs)))

    y_score_no_act_gt = np.zeros((len(test_set), len(test_set.adverbs)))
    y_score_antonym_no_act_gt = np.zeros((len(test_set), len(test_set.adverbs)))

    for idx, data in tqdm(enumerate(data_loader), total=len(data_loader)):
        if args.gpu:
            data[:-1] = [d.cuda() for d in data[:-1]]
        _, predictions, loss_dict = model(data)
        adverb_gt, action_gt = data[1], data[2]

        if isinstance(predictions, dict):
            # faster evaluation, assumes action gt
            use_no_act_gt = False
        else:
            use_no_act_gt = True
            scores_no_act_gt_ant = evaluator.get_antonym_scores(predictions, adverb_gt)
        (
            _,
            action_gt_scores,
            antonym_action_gt_scores,
            _,
        ) = evaluator.get_scores(predictions, action_gt, adverb_gt, use_no_act_gt)

        {
            loss_logging_dict[k].update(v.numpy(), n=data[0].shape[0])
            for k, v in loss_dict.items()
        }

        for j in range(0, data[0].shape[0]):
            # (len_testset, num_adverb)
            y_true_adverb[idx * args.batch_size + j][adverb_gt[j].item()] = 1

            y_score[idx * args.batch_size + j] = action_gt_scores[j][
                test_set.pairidx2idx_array[:, action_gt[j]]
            ].numpy()

            y_score_antonym[idx * args.batch_size + j] = antonym_action_gt_scores[j][
                test_set.pairidx2idx_array[:, action_gt[j]]
            ].numpy()

            if use_no_act_gt:
                for ia, a in enumerate(test_set.adverbs):
                    mask = test_set.get_action_with_adverb_mask(a)
                    y_score_no_act_gt[idx * args.batch_size + j, ia] = predictions[
                        j, mask
                    ].max()
                    y_score_antonym_no_act_gt[
                        idx * args.batch_size + j, ia
                    ] = scores_no_act_gt_ant[j, mask].max()

    v2a_ant = (
        np.argmax(y_true_adverb, axis=1) == np.argmax(y_score_antonym, axis=1)
    ).mean()
    per_adverb = {}
    for adv in test_set.adverb2idx.keys():
        inds = np.where(y_true_adverb[:, test_set.adverb2idx[adv]] == 1)
        per_ant = (
            np.argmax(y_true_adverb[inds], axis=1)
            == np.argmax(y_score_antonym[inds], axis=1)
        ).mean()
        per_adverb[adv] = {"ant": per_ant}
    v2a_ant_cls = sum([per_adverb[adv]["ant"] for adv in per_adverb.keys()]) / len(
        per_adverb.keys()
    )

    a2v_all = average_precision_score(y_true_adverb, y_score)
    a2v_all_w = average_precision_score(y_true_adverb, y_score, average="weighted")

    test_metrics["Acc-A"] = v2a_ant
    test_metrics["mAP-M"] = a2v_all
    test_metrics["mAP-W"] = a2v_all_w
    test_metrics["Acc-A (cls)"] = v2a_ant_cls

    if use_no_act_gt:
        v2a_ant_no_act_gt = (
            np.argmax(y_true_adverb, axis=1)
            == np.argmax(y_score_antonym_no_act_gt, axis=1)
        ).mean()
        a2v_all_no_act_gt = average_precision_score(y_true_adverb, y_score_no_act_gt)
        a2v_all_w_no_act_gt = average_precision_score(
            y_true_adverb, y_score_no_act_gt, average="weighted"
        )
        test_metrics["mAP-M (no act.)"] = a2v_all_no_act_gt
        test_metrics["mAP-W (no act.)"] = a2v_all_w_no_act_gt
        test_metrics["Acc-A (no act.)"] = v2a_ant_no_act_gt

    return test_metrics, loss_logging_dict


def setup_data(args):
    _, test_loader = setup_s3d_data(args)
    test_set = test_loader.dataset

    return test_set, test_loader


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(args: DictConfig):
    assert (
        args.checkpoint is not None
    ), "Please specify a checkpoint folder or file to load from."
    logger.info(f"Run configuration: \n{OmegaConf.to_yaml(args)}")
    logger.info(f"Git commit hash: {get_git_revision_hash(cwd=get_original_cwd())}")

    test_set, test_loader = setup_data(args)
    model = get_model(args.model, test_set).cuda()
    evaluator = Evaluator(test_set)

    checkpoint_path = to_absolute_path(args.checkpoint)

    if os.path.isfile(checkpoint_path) and checkpoint_path.endswith(".pt"):
        checkpoints = [checkpoint_path]
    elif os.path.isdir(checkpoint_path):
        checkpoints = sorted(
            [
                os.path.join(checkpoint_path, f)
                for f in os.listdir(checkpoint_path)
                if f.endswith(".pt") and "ckpt_best" in f
            ]
        )
    else:
        raise ValueError(f"Invalid checkpoint path: {checkpoint_path}")

    test_metrics_list = []
    for checkpoint in checkpoints:
        checkpoint_state = torch.load(checkpoint)
        if "net" in checkpoint_state:
            # legacy
            model.load_state_dict(checkpoint_state["net"])
        else:
            model.load_state_dict(checkpoint_state["model_state"])
        logger.info(f"Loaded model from {str(os.path.basename(checkpoint_path))}")

        test_metrics, _ = test(args, model, test_loader, test_set, evaluator)
        test_metrics_list.append(test_metrics)

    if checkpoint_state["args"].composition == "seen":
        result_print = ""
        result_print_no_act_gt = ""
        for checkpoint, tm in zip(checkpoints, test_metrics_list):
            checkpoint_type = os.path.basename(checkpoint).split(".")[0].split("_")[-1]
            result_print = f"{metric_string_mapping[checkpoint_type]}: {tm[metric_string_mapping[checkpoint_type]]:.3f} | {result_print}"
            if "noact" in checkpoint_type:
                result_print_no_act_gt = f"{metric_string_mapping[checkpoint_type]}: {tm[metric_string_mapping[checkpoint_type]]:.3f} | {result_print_no_act_gt}"
        result_print = f"Test | {result_print}"
        result_print_no_act_gt = f"Test (no action gt) | {result_print_no_act_gt}"
        logger.info(result_print)
        if "map-W (no act.)" in test_metrics.keys():
            logger.info(result_print_no_act_gt)
    else:
        # unseen compositions
        logger.info(f"Test | Acc-A (cls): {100*test_metrics['Acc-A (cls)']:.1f}")


if __name__ == "__main__":
    main()
