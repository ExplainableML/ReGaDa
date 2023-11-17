# Dataset class for pre-extracted S3D features, adapted from https://github.com/dmoltisanti/air-cvpr23
import logging
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from hydra.utils import to_absolute_path
from torch.nn.utils.rnn import pack_sequence, pad_packed_sequence

logger = logging.getLogger(__name__)


def collate_variable_length_seq(
    batch,
    padding_value=0,
    is_train=True,
):
    data = []
    labels = {}
    metadata = {}

    def batchify(d, l, m, data, labels, metadata):
        data.append(d["s3d_features"])

        for stuff, stuff_batch in zip((labels, metadata), (l, m)):
            if len(stuff) == 0:
                for k in stuff_batch.keys():
                    stuff[k] = []

            for k, v in stuff_batch.items():
                stuff[k].append(v)
        return data, labels, metadata

    def padding(data):
        if isinstance(data[0], dict):
            keys = data[0].keys()
            padded_sequence = {}

            for k in keys:
                l = [x[k] for x in data]
                seq = pack_sequence(l, enforce_sorted=False)
                padded_sequence[k], orig_seq_len = pad_packed_sequence(
                    seq, batch_first=True, padding_value=padding_value
                )
        else:
            sequence = pack_sequence(data, enforce_sorted=False)
            padded_sequence, orig_seq_len = pad_packed_sequence(
                sequence, batch_first=True, padding_value=padding_value
            )

        max_len = padded_sequence.shape[1]
        pad_len = max_len - orig_seq_len

        return padded_sequence, pad_len

    for item in batch:
        d, l, m = item
        data, labels, metadata = batchify(d, l, m, data, labels, metadata)

    padded_sequence, pad_len = padding(data)
    labels = {k: torch.LongTensor(v) for k, v in labels.items()}
    negative_adverbs = torch.LongTensor(metadata["negative_adverb"])
    negative_actions = torch.LongTensor(metadata["negative_action"])
    return_data = [
        padded_sequence,
        labels["adverb"],
        labels["verb"],
        labels["pair"],
        pad_len,
        negative_adverbs,
        negative_actions,
    ]

    if not is_train:
        return_data.append(metadata["clip_id"])

    return return_data


def get_verbs_adverbs_pairs(train_df, test_df):
    df = pd.concat([train_df, test_df])
    adverbs = df.clustered_adverb.value_counts().index.to_list()  # sorted by frequency
    verbs = df.clustered_action.value_counts().index.to_list()
    adverb2idx = {a: i for i, a in enumerate(adverbs)}
    idx2adverb = {i: a for i, a, in enumerate(adverbs)}
    verb2idx = {a: i for i, a in enumerate(verbs)}
    idx2verb = {i: a for i, a in enumerate(verbs)}
    pairs = []

    for a in adverbs:
        for v in verbs:
            pairs.append((a, v))

    dataset_data = {
        "adverbs": adverbs,
        "verbs": verbs,
        "adverb2idx": adverb2idx,
        "idx2adverb": idx2adverb,
        "verb2idx": verb2idx,
        "idx2verb": idx2verb,
        "pairs": pairs,
    }

    return dataset_data


class S3DDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df,
        antonyms_df,
        features_dict,
        dataset_data,
        feature_dim,
        feature_dir,
        phase,
        features_dict_merged,
        use_merged_features,
    ):
        self.phase = phase
        self.df = df
        self.random_generator = np.random.default_rng()
        self.antonyms = {t.adverb: t.antonym for t in antonyms_df.itertuples()}

        self.features = features_dict["features"]
        self.features_merged = features_dict_merged["features"]
        self.metadata_merged = features_dict_merged["metadata"]
        self.use_merged_features = use_merged_features
        self.metadata = features_dict["metadata"]
        self.feature_dim = feature_dim

        self.adverbs = dataset_data["adverbs"]
        self.actions = dataset_data["verbs"]
        self.pairs = dataset_data["pairs"]
        self.adverb2idx = dataset_data["adverb2idx"]
        self.action2idx = dataset_data["verb2idx"]
        self.idx2action = dataset_data["idx2verb"]
        self.idx2adverb = dataset_data["idx2adverb"]

        self.pairidx2idx_array = (
            np.zeros((len(self.adverbs), len(self.actions)), dtype=np.short) - 1
        )
        for idx, pair_orig in enumerate(self.pairs):
            self.pairidx2idx_array[
                self.adverb2idx[pair_orig[0].strip()],
                self.action2idx[pair_orig[1].strip()],
            ] = idx

        self.feature_dir = feature_dir

    def __len__(self):
        return len(self.df)

    def get_verb_adv_pair_idx(self, labels):
        v_str = [
            self.idx2action[x.item() if isinstance(x, torch.Tensor) else x]
            for x in labels["verb"]
        ]
        a_str = [
            self.idx2adverb[x.item() if isinstance(x, torch.Tensor) else x]
            for x in labels["adverb"]
        ]
        va_idx = [self.pairs.index((a, v)) for a, v in zip(a_str, v_str)]
        return va_idx

    def get_adverb_with_verb(self, verb):
        verb = verb.item() if isinstance(verb, torch.Tensor) else verb
        verb_str = verb if isinstance(verb, str) else self.idx2action[verb]
        return [a for a, v in self.pairs if v == verb_str]

    def get_action_with_adverb_mask(self, adverb):
        adverb = adverb.item() if isinstance(adverb, torch.Tensor) else adverb
        adverb_str = adverb if isinstance(adverb, str) else self.idx2adverb[adverb]
        return [a == adverb_str for a, v in self.pairs]

    def sample_negative_action(self, action):
        new_action = np.random.randint(len(self.actions))
        if new_action == action:
            return self.sample_negative_action(action)
        return new_action

    def get_data(self, segment):
        adverb_label = self.adverb2idx[segment.clustered_adverb]
        verb_label = self.action2idx[segment.clustered_action]
        labels = dict(
            verb=verb_label,
            adverb=adverb_label,
            pair=self.pairidx2idx_array[adverb_label, verb_label],
        )
        metadata = {
            k: getattr(segment, k)
            for k in (
                "clip_id",
                "start_time",
                "end_time",
                "clustered_adverb",
                "clustered_action",
            )
            if hasattr(segment, k)
        }
        return labels, metadata

    def get_negatives(self, labels, metadata):
        adverb = metadata["clustered_adverb"]
        verb_label = labels["verb"]
        adverb_label = labels["adverb"]
        # sample negative adverbs
        neg_adverb = self.adverb2idx[self.antonyms[adverb]]
        assert adverb_label != neg_adverb
        neg_action = self.sample_negative_action(verb_label)

        return neg_adverb, neg_action

    def __getitem__(self, item):
        if isinstance(item, torch.Tensor):
            item = item.item()

        segment = self.df.iloc[item]
        labels, metadata = self.get_data(segment)
        metadata["negative_adverb"], metadata["negative_action"] = self.get_negatives(
            labels, metadata
        )

        data, frame_samples = self.load_features(segment)
        metadata["frame_samples"] = frame_samples

        return data, labels, metadata

    def load_features(self, segment):
        uid = self.get_seg_id(segment)
        if self.use_merged_features:
            features = self.features_merged[uid]
            frame_samples = self.metadata_merged[uid]["frame_samples"].squeeze()
        else:
            features = self.features[uid]
            frame_samples = self.metadata[uid]["frame_samples"].squeeze()
        return features, frame_samples

    @staticmethod
    def get_seg_id(segment):
        seg_id = (
            segment["clip_id"]
            if isinstance(segment, dict)
            else getattr(segment, "clip_id")
        )
        return seg_id


def load_features(feature_dir, set_):
    feature_dict = {}
    features_path = (
        Path(feature_dir) / "ht1m_init/stack_size=16_freq=1/rgb/" / f"{set_}.pth"
    )
    logger.info(f"Loading features from {features_path}")
    fd = torch.load(features_path, map_location=torch.device("cpu"))
    features = fd["features"]

    for k, v in features.items():
        if isinstance(v, torch.Tensor):
            features[k] = v
        else:
            assert isinstance(v, dict)
            features[k] = {kk: vv for kk, vv in v.items()}

    feature_dict["features"] = features
    feature_dict["metadata"] = fd["metadata"]

    return feature_dict


def merge_s3d_features(train_features, test_features):
    features_merged = {}
    features_merged["features"] = {
        **train_features["features"],
        **test_features["features"],
    }
    features_merged["metadata"] = {
        **train_features["metadata"],
        **test_features["metadata"],
    }
    return features_merged


def setup_s3d_data(args):
    data_dir = to_absolute_path(args.data_dir)

    train_df = pd.read_csv(Path(data_dir) / "train.csv")
    test_df = pd.read_csv(Path(data_dir) / "test.csv")

    antonyms_df = pd.read_csv(Path(data_dir) / "antonyms.csv")
    dataset_data = get_verbs_adverbs_pairs(train_df, test_df)

    feature_dim = 1024
    feature_dir_train = Path(to_absolute_path(args.train_feature_dir))
    feature_dir_test = Path(to_absolute_path(args.test_feature_dir))

    collate_fn_train = partial(
        collate_variable_length_seq,
        is_train=True,
    )
    collate_fn_test = partial(
        collate_variable_length_seq,
        is_train=False,
    )

    features_train = load_features(feature_dir_train, "train")
    features_test = load_features(feature_dir_test, "test")

    if args.s3d_use_merged_features:
        # merging of data sources required for unseen compositions
        # train/test splits contain data from both train and test of original splits
        features_merged = merge_s3d_features(features_train, features_test)
    else:
        features_merged = {"features": None, "metadata": None}

    train_dataset = S3DDataset(
        train_df,
        antonyms_df,
        features_train,
        dataset_data,
        feature_dim,
        feature_dir_train,
        phase="train",
        features_dict_merged=features_merged,
        use_merged_features=args.s3d_use_merged_features,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=collate_fn_train,
    )

    test_dataset = S3DDataset(
        test_df,
        antonyms_df,
        features_test,
        dataset_data,
        feature_dim,
        feature_dir_test,
        phase="test",
        features_dict_merged=features_merged,
        use_merged_features=args.s3d_use_merged_features,
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        num_workers=args.workers,
        collate_fn=collate_fn_test,
    )

    return train_loader, test_loader
