import gc
import logging
import os
from collections import defaultdict
from pathlib import Path
from test import test

import hydra
import torch
from hydra.utils import get_original_cwd
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from dataset import setup_s3d_data
from models import get_model
from models.evaluator import Evaluator
from tb_logger import TbLogger
from utils import (
    AverageMeter,
    get_git_revision_hash,
    has_improved,
    metric_defaultdict,
    metric_string_mapping,
    save_checkpoint,
    setup_seed,
)

logger = logging.getLogger(__name__)
OmegaConf.register_new_resolver(
    "basename", lambda x: os.path.basename(os.path.abspath(x)), replace=True
)


@hydra.main(config_path="config", config_name="default", version_base="1.2")
def main(args: DictConfig):
    logger.info(f"Run configuration: \n{OmegaConf.to_yaml(args)}")
    logger.info(f"Git commit hash: {get_git_revision_hash(cwd=get_original_cwd())}")

    if args.cuda_device:
        logger.info(f"Setting cuda device to device {args.cuda_device}!")
        torch.cuda.set_device(int(args.cuda_device))

    if args.seed >= 0:
        setup_seed(args.seed)
        logger.info(f"Seeding enabled! Seeding experiment with SEED = {args.seed}")

    datasets, data_loaders = setup_data(args)
    train_set, val_set, _ = datasets
    train_loader, val_loader, _ = data_loaders

    model = get_model(args.model, train_set).cuda()

    evaluator = Evaluator(train_set)

    params = [param for name, param in model.named_parameters() if param.requires_grad]
    optim_params = [
        {"name": "embedding", "params": params},
    ]
    optimizer = hydra.utils.instantiate(
        config=args.optimizer, params=optim_params, _convert_="all"
    )
    logger.info(f"Using optimizer '{args.optimizer._target_}'")

    start_epoch = 0
    best_metrics = metric_defaultdict()

    checkpoint_path = None
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint).absolute()

    if checkpoint_path is not None:
        logger.info(
            f"Loading and resuming training from checkpoint '{checkpoint_path}'"
        )
        (
            model,
            optimizer,
            start_epoch,
            best_metrics,
        ) = load_from_checkpoint(str(checkpoint_path.as_posix()), model, optimizer)

    tb_logger = TbLogger("logs")

    epoch = 0
    for epoch in range(start_epoch, args.max_epochs + 1):
        train_loss = train(model, train_loader, optimizer, tb_logger, epoch)
        logger.info(f"Epoch: {epoch} | Train Loss: {train_loss:.2E}")

        if epoch % args.eval_interval == 0:
            test_metrics, loss_dict = test(args, model, val_loader, val_set, evaluator)
            {
                tb_logger.log_metrics(
                    f"Loss{val_set.phase.capitalize()}/{k}", v.avg, epoch
                )
                for k, v in loss_dict.items()
            }

            for k, v in test_metrics.items():
                if has_improved(best_metrics[k], v, k):
                    best_metrics[k] = v
                    # save checkpoints for best metrics during training
                    for save_metric in args.save_metrics:
                        if k == metric_string_mapping[save_metric]:
                            save_checkpoint(
                                model,
                                optimizer,
                                args,
                                epoch,
                                best_metrics,
                                save_metric,
                            )
            logger.info(f"Test | Test Loss: {loss_dict['loss_total'].avg:.2E}")

    if args.composition == "seen":
        logger.info(
            f"Test | mAP-W: {best_metrics['mAP-W']:.3f} | "
            f"mAP-M: {best_metrics['mAP-M']:.3f} | "
            f"Acc-A: {best_metrics['Acc-A']:.3f}"
        )
        if "mAP-W (no act.)" in best_metrics.keys():
            logger.info(
                f"Test (no action gt) | mAP-W: {best_metrics['mAP-W (no act.)']:.3f} | "
                f"mAP-M: {best_metrics['mAP-M (no act.)']:.3f} | "
                f"Acc-A: {best_metrics['Acc-A (no act.)']:.3f}"
            )
    else:
        # unseen compositions
        logger.info(f"Test | Acc: {100*best_metrics['Acc-A (cls)']:.1f}")

    # log best metrics
    best_metrics = {f"Test/{k}": v for k, v in best_metrics.items()}
    tb_logger.log_hparams(args, best_metrics)

    tb_logger.close()


def setup_data(args):
    train_loader, test_loader = setup_s3d_data(args)
    val_loader = test_loader
    train_set = train_loader.dataset
    val_set = val_loader.dataset
    test_set = test_loader.dataset
    return (train_set, val_set, test_set), (train_loader, val_loader, test_loader)


def load_from_checkpoint(checkpoint_path, model, optimizer):
    checkpoint_state = torch.load(checkpoint_path)
    pretrained_state_dict = checkpoint_state["model_state"]
    model_state_dict = model.state_dict()
    pretrained_state_dict = {
        k: v for k, v in pretrained_state_dict.items() if k in model_state_dict
    }
    model_state_dict.update(pretrained_state_dict)
    model.load_state_dict(model_state_dict)
    start_epoch = checkpoint_state["epoch"] + 1
    current_best_score = checkpoint_state["best_score"]
    if "optimizer_state" in checkpoint_state.keys():
        optimizer.load_state_dict(checkpoint_state["optimizer_state"])

    return (model, optimizer, start_epoch, current_best_score)


def train(
    model,
    train_loader,
    optimizer,
    tb_logger,
    epoch,
):
    model.train()
    loss_logging_dict = defaultdict(AverageMeter)
    train_loss_avg = AverageMeter()

    for idx, data in tqdm(enumerate(train_loader), total=len(train_loader)):
        data = [d.cuda() for d in data]
        model_result = model(data)

        train_loss, loss_dict = model_result[0], model_result[2]

        {
            loss_logging_dict[k].update(v.numpy(), n=data[0].shape[0])
            for k, v in loss_dict.items()
        }
        # optimizer step
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        train_loss_avg.update(train_loss.detach().cpu().numpy())

    {
        tb_logger.log_metrics(f"LossTrain/{k}", v.avg, epoch)
        for k, v in loss_logging_dict.items()
    }

    gc.collect()
    return train_loss_avg.avg


if __name__ == "__main__":
    main()
