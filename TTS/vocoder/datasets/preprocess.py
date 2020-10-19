import glob
import os
from pathlib import Path
from tqdm import tqdm

import numpy as np


def preprocess_wav_files(out_path, config, ap):
    os.makedirs(os.path.join(out_path, "quant"), exist_ok=True)
    os.makedirs(os.path.join(out_path, "mel"), exist_ok=True)
    wav_files = find_wav_files(config.data_path)
    for path in tqdm(wav_files):
        wav_name = Path(path).stem
        quant_path = os.path.join(out_path, "quant", wav_name + ".npy")
        mel_path = os.path.join(out_path, "mel", wav_name + ".npy")
        y = ap.load_wav(path)
        mel = ap.melspectrogram(y)
        np.save(mel_path, mel)
        if isinstance(config.mode, int):
            quant = (
                ap.mulaw_encode(y, qc=config.mode)
                if config.mulaw
                else ap.quantize(y, bits=config.mode)
            )
            np.save(quant_path, quant)


def find_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, "**", "*.wav"), recursive=True)
    return wav_paths


def find_feat_files(data_path):
    feat_paths = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)
    return feat_paths


def load_wav_data(data_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    np.random.seed(0)
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def load_wav_feat_data(data_path, feat_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    feat_paths = find_feat_files(feat_path)

    wav_paths.sort(key=lambda x: Path(x).stem)
    feat_paths.sort(key=lambda x: Path(x).stem)

    assert len(wav_paths) == len(feat_paths)
    for wav, feat in zip(wav_paths, feat_paths):
        wav_name = Path(wav).stem
        feat_name = Path(feat).stem
        assert wav_name == feat_name

    items = list(zip(wav_paths, feat_paths))
    np.random.seed(0)
    np.random.shuffle(items)
    return items[:eval_split_size], items[eval_split_size:]
