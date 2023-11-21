import glob
import os
from pathlib import Path

import numpy as np
from coqpit import Coqpit
from tqdm import tqdm

from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import mulaw_encode, quantize


def preprocess_wav_files(out_path: str, config: Coqpit, ap: AudioProcessor):
    """Process wav and compute mel and quantized wave signal.
    It is mainly used by WaveRNN dataloader.

    Args:
        out_path (str): Parent folder path to save the files.
        config (Coqpit): Model config.
        ap (AudioProcessor): Audio processor.
    """
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
                mulaw_encode(wav=y, mulaw_qc=config.mode)
                if config.model_args.mulaw
                else quantize(x=y, quantize_bits=config.mode)
            )
            np.save(quant_path, quant)


def find_wav_files(data_path, file_ext="wav"):
    wav_paths = glob.glob(os.path.join(data_path, "**", f"*.{file_ext}"), recursive=True)
    return wav_paths


def find_feat_files(data_path):
    feat_paths = glob.glob(os.path.join(data_path, "**", "*.npy"), recursive=True)
    return feat_paths


def load_wav_data(data_path, eval_split_size, file_ext="wav"):
    wav_paths = find_wav_files(data_path, file_ext=file_ext)
    assert len(wav_paths) > 0, f" [!] {data_path} is empty."
    np.random.seed(0)
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]


def load_wav_feat_data(data_path, feat_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    feat_paths = find_feat_files(feat_path)

    wav_paths.sort(key=lambda x: Path(x).stem)
    feat_paths.sort(key=lambda x: Path(x).stem)

    assert len(wav_paths) == len(feat_paths), f" [!] {len(wav_paths)} vs {feat_paths}"
    for wav, feat in zip(wav_paths, feat_paths):
        wav_name = Path(wav).stem
        feat_name = Path(feat).stem
        assert wav_name == feat_name

    items = list(zip(wav_paths, feat_paths))
    np.random.seed(0)
    np.random.shuffle(items)
    return items[:eval_split_size], items[eval_split_size:]
