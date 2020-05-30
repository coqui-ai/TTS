import glob
import os

import numpy as np


def find_wav_files(data_path):
    wav_paths = glob.glob(os.path.join(data_path, '**', '*.wav'), recursive=True)
    return wav_paths


def load_wav_data(data_path, eval_split_size):
    wav_paths = find_wav_files(data_path)
    np.random.seed(0)
    np.random.shuffle(wav_paths)
    return wav_paths[:eval_split_size], wav_paths[eval_split_size:]
