import glob
import os
from pathlib import Path

import numpy as np


def find_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    return wav_paths


def find_feat_files(data_path):
    feat_paths = glob.glob(os.path.join(data_path, '**', '*.npy'), recursive=True)
    return feat_paths


def load_wav_data(data_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    np.random.seed(0)
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def load_wav_feat_data(data_path, feat_path, eval_split_size):
    wav_paths = sorted(find_wav_files(data_path))
    feat_paths = sorted(find_feat_files(feat_path))
    assert len(wav_paths) == len(feat_paths)
    for wav, feat in zip(wav_paths, feat_paths):
        wav_name = Path(wav).stem
        feat_name = Path(feat).stem
        assert wav_name == feat_name

    items = list(zip(wav_paths, feat_paths))
    np.random.seed(0)
    np.random.shuffle(items)
    return items[:eval_split_size], items[eval_split_size:]
