import logging
import os
import random
import subprocess
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


class MyDict(dict):
    __missing__ = lambda self, key: key


class metric_defaultdict(defaultdict):
    def __init__(self):
        super().__init__(None)  # base class doesn't get a factory
        self.f_of_x = (
            lambda key: float("inf") if "loss" in key else 0.0
        )  # f_of_x  # save f(x)

    def __missing__(self, key):  # called when a default needed
        ret = self.f_of_x(key)  # calculate default value
        self[key] = ret  # and install it in the dict
        return ret


metric_string_mapping = MyDict(
    {
        "acc-a": "Acc-A",
        "map-m": "mAP-M",
        "map-w": "mAP-W",
        "acc-a-cls": "Acc-A (cls)",
        "map-m-noact": "mAP-M (no act.)",
        "map-w-noact": "mAP-W (no act.)",
        "acc-a-noact": "Acc-A (no act.)",
    }
)


def has_improved(best, current, metric_name):
    if "loss" in metric_name.lower():
        return best > current
    else:
        return best < current


def setup_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def get_git_revision_hash(cwd):
    try:
        hash_string = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd)
            .decode("ascii")
            .strip()
        )
    except:
        hash_string = ""
    return hash_string


def save_checkpoint(
    model,
    optimizer,
    args,
    epoch,
    best_score,
    save_metric,
):
    state = {
        "args": args,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "best_score": best_score,
        "epoch": epoch,
    }

    checkpoint_file = f"ckpt_best_{save_metric}.pt"
    torch.save(state, checkpoint_file)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
