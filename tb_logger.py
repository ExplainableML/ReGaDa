from collections import MutableMapping
from copy import deepcopy

from omegaconf import OmegaConf
from torch.utils.tensorboard import SummaryWriter


class TbLogger(object):
    def __init__(self, log_dir: str):
        self.tb_writer = SummaryWriter(log_dir=log_dir, flush_secs=30)

    def log_metrics(self, key, value, global_step):
        self.tb_writer.add_scalar(key, value, global_step)

    def log_hparams(self, args, metrics):
        copy_args = deepcopy(args)
        copy_args = OmegaConf.to_container(copy_args, resolve=True)
        args_dict = flatten_dict(copy_args)
        self.tb_writer.add_hparams(
            hparam_dict=args_dict, metric_dict=dict(metrics), run_name="eval"
        )

    def log_figure(self, tag, figure, global_step):
        self.tb_writer.add_figure(tag, figure, global_step)

    def log_histogram(self, tag, array, global_step, bins="tensorflow"):
        self.tb_writer.add_histogram(tag, array, global_step, bins)

    def log_text(self, tag, text, global_step):
        self.tb_writer.add_text(tag, text, global_step)

    def close(self):
        self.tb_writer.close()


def flatten_dict(d, parent_key="", sep="."):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            for i, v_list in enumerate(v):
                items.append((new_key + sep + str(i), v_list))
        else:
            items.append((new_key, v))
    return dict(items)
