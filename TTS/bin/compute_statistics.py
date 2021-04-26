#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

from TTS.tts.datasets.preprocess import load_meta_data
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config


def main():
    """Compute feature statistics for normalization."""
    parser = argparse.ArgumentParser(description="Compute mean and variance of spectrogtram features.")
    parser.add_argument(
        "--config_path", type=str, required=True, help="TTS config file path to define audio processing parameters."
    )
    parser.add_argument(
        "--stats_path",
        type=str,
        help=("Save path (directory and filename). " "If not specified taken from config.json."),
    )
    args = parser.parse_args()

    # load config
    c = load_config(args.config_path)
    c.audio["signal_norm"] = False  # do not apply earlier normalization
    if args.stats_path is None:
        output_file_path = c.audio["stats_path"]
    else:
        output_file_path = args.stats_path
    c.audio["stats_path"] = None  # discard pre-defined stats

    # load audio processor
    ap = AudioProcessor(**c.audio)

    # load the meta data of target dataset
    if "data_path" in c.keys():
        dataset_items = glob.glob(os.path.join(c.data_path, "**", "*.wav"), recursive=True)
    else:
        dataset_items = load_meta_data(c.datasets)[0]  # take only train data
    print(f" > There are {len(dataset_items)} files.")

    mel_sum = 0
    mel_square_sum = 0
    linear_sum = 0
    linear_square_sum = 0
    N = 0
    for item in tqdm(dataset_items, ncols=80):
        # compute features
        wav = ap.load_wav(item if isinstance(item, str) else item[1])
        linear = ap.spectrogram(wav)
        mel = ap.melspectrogram(wav)

        # compute stats
        N += mel.shape[1]
        mel_sum += mel.sum(1)
        linear_sum += linear.sum(1)
        mel_square_sum += (mel ** 2).sum(axis=1)
        linear_square_sum += (linear ** 2).sum(axis=1)

    mel_mean = mel_sum / N
    mel_scale = np.sqrt(mel_square_sum / N - mel_mean ** 2)
    linear_mean = linear_sum / N
    linear_scale = np.sqrt(linear_square_sum / N - linear_mean ** 2)

    stats = {}
    stats["mel_mean"] = mel_mean
    stats["mel_std"] = mel_scale
    stats["linear_mean"] = linear_mean
    stats["linear_std"] = linear_scale

    print(f" > Avg mel spec mean: {mel_mean.mean()}")
    print(f" > Avg mel spec scale: {mel_scale.mean()}")
    print(f" > Avg linear spec mean: {linear_mean.mean()}")
    print(f" > Avg lienar spec scale: {linear_scale.mean()}")

    # set default config values for mean-var scaling
    c.audio["stats_path"] = output_file_path
    c.audio["signal_norm"] = True
    # remove redundant values
    del c.audio["max_norm"]
    del c.audio["min_level_db"]
    del c.audio["symmetric_norm"]
    del c.audio["clip_norm"]
    stats["audio_config"] = c.audio
    print(c.audio)
    np.save(output_file_path, stats, allow_pickle=True)
    print(f" > stats saved to {output_file_path}")


if __name__ == "__main__":
    main()
