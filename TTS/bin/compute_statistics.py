#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

# from TTS.utils.io import load_config
from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor


def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(description="Compute mean and variance of spectrogtram features.")
    parser.add_argument("config_path", type=str, help="TTS config file path to define audio processin parameters.")
    parser.add_argument("out_path", type=str, help="save path (directory and filename).")
    parser.add_argument(
        "--data_path",
        type=str,
        required=False,
        help="folder including the target set of wavs overriding dataset config.",
    )
    args, overrides = parser.parse_known_args()

    CONFIG = load_config(args.config_path)
    CONFIG.parse_known_args(overrides, relaxed_parser=True)

    # load config
    CONFIG.audio.signal_norm = False  # do not apply earlier normalization
    CONFIG.audio.stats_path = None  # discard pre-defined stats

    # load audio processor
    ap = AudioProcessor(**CONFIG.audio.to_dict())

    # load the meta data of target dataset
    if args.data_path:
        dataset_items = glob.glob(os.path.join(args.data_path, "**", "*.wav"), recursive=True)
    else:
        dataset_items = load_tts_samples(CONFIG.datasets)[0]  # take only train data
    print(f" > There are {len(dataset_items)} files.")

    mel_sum = 0
    mel_square_sum = 0
    linear_sum = 0
    linear_square_sum = 0
    N = 0
    for item in tqdm(dataset_items):
        # compute features
        wav = ap.load_wav(item if isinstance(item, str) else item["audio_file"])
        linear = ap.spectrogram(wav)
        mel = ap.melspectrogram(wav)

        # compute stats
        N += mel.shape[1]
        mel_sum += mel.sum(1)
        linear_sum += linear.sum(1)
        mel_square_sum += (mel**2).sum(axis=1)
        linear_square_sum += (linear**2).sum(axis=1)

    mel_mean = mel_sum / N
    mel_scale = np.sqrt(mel_square_sum / N - mel_mean**2)
    linear_mean = linear_sum / N
    linear_scale = np.sqrt(linear_square_sum / N - linear_mean**2)

    output_file_path = args.out_path
    stats = {}
    stats["mel_mean"] = mel_mean
    stats["mel_std"] = mel_scale
    stats["linear_mean"] = linear_mean
    stats["linear_std"] = linear_scale

    print(f" > Avg mel spec mean: {mel_mean.mean()}")
    print(f" > Avg mel spec scale: {mel_scale.mean()}")
    print(f" > Avg linear spec mean: {linear_mean.mean()}")
    print(f" > Avg linear spec scale: {linear_scale.mean()}")

    # set default config values for mean-var scaling
    CONFIG.audio.stats_path = output_file_path
    CONFIG.audio.signal_norm = True
    # remove redundant values
    del CONFIG.audio.max_norm
    del CONFIG.audio.min_level_db
    del CONFIG.audio.symmetric_norm
    del CONFIG.audio.clip_norm
    stats["audio_config"] = CONFIG.audio.to_dict()
    np.save(output_file_path, stats, allow_pickle=True)
    print(f" > stats saved to {output_file_path}")


if __name__ == "__main__":
    main()
