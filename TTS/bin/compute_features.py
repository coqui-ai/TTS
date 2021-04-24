import argparse
import glob
import os
import shutil

import numpy as np
from tqdm import tqdm

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config

parser = argparse.ArgumentParser(
    description='Compute feature vectors for each wav file in a dataset.'
    )
parser.add_argument(
    "--config_path",
    type=str,
    required=True,
    help="Path to config file for feature extraction.",
    )
parser.add_argument(
    "--data_path",
    type=str,
    help=("Data path for wav files - directory or CSV file. "
          "If blank taken from config.json")
    )
parser.add_argument(
    "--feature_path",
    type=str,
    help=("Path to store features. "
          "If blank taken from config.json")
    )
parser.add_argument(
    "--separator",
    type=str,
    help="Separator used in file if CSV is passed for data_path",
    default="|"
    )
args = parser.parse_args()

c = load_config(args.config_path)
ap = AudioProcessor(**c["audio"])

data_path = args.data_path
if data_path is None:
    data_path = c["data_path"]

feature_path = args.feature_path
if feature_path is None:
    feature_path = c["feature_path"]

sep = args.separator

if data_path.lower().endswith(".csv"):
    # Parse CSV
    print(f"CSV file: {data_path}")
    with open(data_path) as f:
        wav_path = os.path.join(os.path.dirname(data_path), "wavs")
        wav_files = []
        print(f"Separator is: {sep}")
        for line in f:
            components = line.split(sep)
            if len(components) != 2:
                print("Invalid line")
                continue
            wav_file = os.path.join(wav_path, components[0] + ".wav")
            if os.path.exists(wav_file):
                wav_files.append(wav_file)
        print(f"Count of wavs imported: {len(wav_files)}")
else:
    # Parse all wav files in data_path
    wav_files = glob.glob(data_path + "/**/*.wav", recursive=True)
    print(f"Count of wavs globbed: {len(wav_files)}")

# make feature dirs
# check if multiple output paths are needed, i.e. wavs were in nested folders
feature_paths = [
    os.path.dirname(wav_file).replace(data_path, feature_path)
    for wav_file in wav_files
    ]
feature_paths = set(feature_paths)

for path in feature_paths:
    os.makedirs(path, exist_ok=True)

feature_files = [
    wav_file.replace(data_path, feature_path).replace(".wav", ".npy")
    for wav_file in wav_files
    ]

# compute features
for wav_file, feature_file in tqdm(
        zip(wav_files, feature_files), total=len(wav_files), ncols=80
        ):
    mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=ap.sample_rate))
    np.save(feature_file, mel_spec)
print(f" > features saved to {feature_path}")

# save audio config for checking
# remove redundant values
del c.audio["max_norm"]
del c.audio["min_level_db"]
del c.audio["symmetric_norm"]
del c.audio["clip_norm"]
audio_config = {"audio_config": c.audio}
print(c.audio)
config_name = 'feats_audio_config.npy'
# save it to the parent folder of the feats directory
np.save(f'{feature_path}/../{config_name}', audio_config)
print(f" > audio config saved in feature dir as {config_name}")
