#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import librosa
import yaml
import shutil
import argparse
import matplotlib.pyplot as plt
import math, pickle, os, glob
import numpy as np
from tqdm import tqdm
from TTS.tts.utils.audio import AudioProcessor
from TTS.tts.utils.generic_utils import load_config
from multiprocessing import Pool

os.environ["OMP_NUM_THREADS"] = "1"

def get_files(path, extension=".wav"):
    filenames = []
    for filename in glob.iglob(f"{path}/**/*{extension}", recursive=True):
        filenames += [filename]
    return filenames


def _process_file(path):
    wav = ap.load_wav(path)
    mel = ap.melspectrogram(wav)
    wav = wav.astype(np.float32)
    # check
    assert len(wav.shape) == 1, \
        f"{path} seems to be multi-channel signal."
    assert np.abs(wav).max() <= 1.0, \
        f"{path} seems to be different from 16 bit PCM."

    # gap when wav is not multiple of hop_length
    gap = wav.shape[0] % ap.hop_length
    assert mel.shape[1] * ap.hop_length == wav.shape[0] + ap.hop_length - gap, f'{mel.shape[1] * ap.hop_length} vs {wav.shape[0] + ap.hop_length + gap}'
    return mel.astype(np.float32), wav


def extract_feats(wav_path):
    idx = wav_path.split("/")[-1][:-4]
    m, wav = _process_file(wav_path)
    mel_path = f"{MEL_PATH}{idx}.npy"
    np.save(mel_path, m.astype(np.float32), allow_pickle=False)
    return wav_path, mel_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config_path", type=str, help="path to config file for feature extraction."
    )
    parser.add_argument(
        "--num_procs", type=int, default=4, help="number of parallel processes."
    )
    parser.add_argument(
        "--data_path", type=str, default='', help="path to audio files."
    )
    parser.add_argument(
        "--out_path", type=str, default='', help="destination to write files."
    )
    parser.add_argument(
        "--ignore_errors", type=bool, default=False, help="ignore bad files."
    )
    args = parser.parse_args()

    # load config
    config = load_config(args.config_path)
    config.update(vars(args))

    config.audio['do_trim_silence'] = False
    # config['audio']['signal_norm'] = False  # do not apply earlier normalization

    ap = AudioProcessor(**config['audio'])

    SEG_PATH = config['data_path']
    OUT_PATH = args.out_path
    MEL_PATH = os.path.join(OUT_PATH, "mel/")
    os.makedirs(OUT_PATH, exist_ok=True)
    os.makedirs(MEL_PATH, exist_ok=True)

    # TODO: use TTS data processors
    wav_files = get_files(SEG_PATH)
    print(" > Number of audio files : {}".format(len(wav_files)))

    wav_file = wav_files[0]
    m, wav = _process_file(wav_file)

    # sanity check
    print(' > Sample Spec Stats...')
    print(' | > spectrogram max:', m.max())
    print(' | > spectrogram min: ', m.min())
    print(' | > spectrogram shape:', m.shape)
    print(' | > wav shape:', wav.shape)
    print(' | > wav max - min:', wav.max(), ' - ', wav.min())

    # This will take a while depending on size of dataset
    #with Pool(args.num_procs) as p:
    #    dataset_ids = list(tqdm(p.imap(extract_feats, wav_files), total=len(wav_files)))
    dataset_ids = []
    for wav_file in tqdm(wav_files):
         item_id = extract_feats(wav_file)
         dataset_ids.append(item_id)

    # save metadata
    with open(os.path.join(OUT_PATH, "metadata.txt"), "w") as f:
        for data in dataset_ids:
            f.write(f"{data[0]}|{data[1]}\n")
