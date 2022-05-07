import bisect

import numpy as np
import torch


def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=_pad)


def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_tensor(x, length):
    _pad = 0.0
    assert x.ndim == 2
    x = np.pad(x, [[0, 0], [0, length - x.shape[1]]], mode="constant", constant_values=_pad)
    return x


def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[1] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])


def _pad_stop_target(x: np.ndarray, length: int, pad_val=1) -> np.ndarray:
    """Pad stop target array.

    Args:
        x (np.ndarray): Stop target array.
        length (int): Length after padding.
        pad_val (int, optional): Padding value. Defaults to 1.

    Returns:
        np.ndarray: Padded stop target array.
    """
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=pad_val)


def prepare_stop_target(inputs, out_steps):
    """Pad row vectors with 1."""
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])


def pad_per_step(inputs, pad_len):
    return np.pad(inputs, [[0, 0], [0, 0], [0, pad_len]], mode="constant", constant_values=0.0)


def get_length_balancer_weights(items: list, num_buckets=10):
    # get all durations
    audio_lengths = np.array([item["audio_length"] for item in items])
    # create the $num_buckets buckets classes based in the dataset max and min length
    max_length = int(max(audio_lengths))
    min_length = int(min(audio_lengths))
    step = int((max_length - min_length) / num_buckets)
    buckets_classes = [i + step for i in range(min_length, max_length, step)]
    # add each sample in their respective length bucket
    buckets_names = np.array(
        [buckets_classes[bisect.bisect_left(buckets_classes, item["audio_length"])] for item in items]
    )
    # count and compute the weights_bucket for each sample
    unique_buckets_names = np.unique(buckets_names).tolist()
    bucket_ids = [unique_buckets_names.index(l) for l in buckets_names]
    bucket_count = np.array([len(np.where(buckets_names == l)[0]) for l in unique_buckets_names])
    weight_bucket = 1.0 / bucket_count
    dataset_samples_weight = np.array([weight_bucket[l] for l in bucket_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
