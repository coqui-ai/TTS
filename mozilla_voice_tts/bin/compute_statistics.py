#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse

import numpy as np
from tqdm import tqdm

from mozilla_voice_tts.tts.datasets.preprocess import load_meta_data
from mozilla_voice_tts.utils.io import load_config
from mozilla_voice_tts.utils.audio import AudioProcessor

def main():
    """Run preprocessing process."""
    parser = argparse.ArgumentParser(
        description="Compute mean and variance of spectrogtram features.")
    parser.add_argument("--config_path", type=str, required=True,
                        help="TTS config file path to define audio processin parameters.")
    parser.add_argument("--out_path", default=None, type=str,
                        help="directory to save the output file.")
    args = parser.parse_args()

    # load config
    CONFIG = load_config(args.config_path)
    CONFIG.audio['signal_norm'] = False  # do not apply earlier normalization
    CONFIG.audio['stats_path'] = None  # discard pre-defined stats

    # load audio processor
    ap = AudioProcessor(**CONFIG.audio)

    # load the meta data of target dataset
    dataset_items = load_meta_data(CONFIG.datasets)[0]  # take only train data
    print(f" > There are {len(dataset_items)} files.")

    mel_sum = 0
    mel_square_sum = 0
    linear_sum = 0
    linear_square_sum = 0
    N = 0
    for item in tqdm(dataset_items):
        # compute features
        wav = ap.load_wav(item[1])
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

    output_file_path = os.path.join(args.out_path, "scale_stats.npy")
    stats = {}
    stats['mel_mean'] = mel_mean
    stats['mel_std'] = mel_scale
    stats['linear_mean'] = linear_mean
    stats['linear_std'] = linear_scale

    print(f' > Avg mel spec mean: {mel_mean.mean()}')
    print(f' > Avg mel spec scale: {mel_scale.mean()}')
    print(f' > Avg linear spec mean: {linear_mean.mean()}')
    print(f' > Avg lienar spec scale: {linear_scale.mean()}')

    # set default config values for mean-var scaling
    CONFIG.audio['stats_path'] = output_file_path
    CONFIG.audio['signal_norm'] = True
    # remove redundant values
    del CONFIG.audio['max_norm']
    del CONFIG.audio['min_level_db']
    del CONFIG.audio['symmetric_norm']
    del CONFIG.audio['clip_norm']
    stats['audio_config'] = CONFIG.audio
    np.save(output_file_path, stats, allow_pickle=True)
    print(f' > scale_stats.npy is saved to {output_file_path}')


if __name__ == "__main__":
    main()
