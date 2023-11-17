# Code adapted from https://github.com/dmoltisanti/air-cvpr23
import logging
import torch

from models.s3d.s3dg import S3D

logger = logging.getLogger(__name__)


class TextEncoder(object):
    __encoder__ = None

    @classmethod
    def get_text_embeddings(cls, s3d_init_folder, words):
        assert isinstance(words, list)

        if cls.__encoder__ is None:
            logger.info("Creating S3D text encoder")
            init_dict_path = s3d_init_folder / "s3d_dict.npy"
            s3d_checkpoint_path = s3d_init_folder / "s3d_howto100m.pth"
            logger.info(f"Loading S3D weights from {s3d_init_folder}")
            original_classes = 512
            s3d = S3D(init_dict_path, num_classes=original_classes)
            s3d.load_state_dict(torch.load(s3d_checkpoint_path))
            encoder = s3d.text_module
            encoder = encoder.cuda()
            cls.__encoder__ = encoder

        embeddings = cls.__encoder__(words, to_gpu=True)["text_embedding"]

        return embeddings
