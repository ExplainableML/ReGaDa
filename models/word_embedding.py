import logging
from pathlib import Path

from models.s3d.text_encoder import TextEncoder as S3DTextEncoder

logger = logging.getLogger(__name__)


def load_word_embeddings(emb_type, vocab, feature_dir):
    # assumes word embeddings to be in parent folders of features
    feature_dir = Path(feature_dir).parent.parent

    if emb_type == "s3d":
        embeds = load_s3d_embeddings(vocab, feature_dir)
    else:
        raise ValueError("Invalid embedding")
    return embeds


def load_s3d_embeddings(vocab, feature_dir):
    s3d_init_path = Path(feature_dir) / "s3d"
    embeds = S3DTextEncoder.get_text_embeddings(s3d_init_path, vocab)
    return embeds
