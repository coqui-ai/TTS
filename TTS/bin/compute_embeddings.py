import argparse
import glob
import os

import torch
import numpy as np
from tqdm import tqdm

from TTS.config import BaseDatasetConfig, load_config
from TTS.speaker_encoder.utils.generic_utils import setup_model
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

from TTS.config import load_config


parser = argparse.ArgumentParser(
    description='Compute embedding vectors for each wav file in a dataset.'
)
parser.add_argument("model_path", type=str, help="Path to model checkpoint file.")
parser.add_argument(
    "config_path",
    type=str,
    help="Path to model config file.",
)

parser.add_argument(
    "config_dataset_path",
    type=str,
    help="Path to dataset config file.",
)
parser.add_argument("output_path", type=str, help="path for output speakers.json and/or speakers.npy.")
parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
parser.add_argument("--save_npy", type=bool, help="flag to set cuda.", default=False)
args = parser.parse_args()


c = load_config(args.config_path)
c_dataset = load_config(args.config_dataset_path)

ap = AudioProcessor(**c["audio"])

train_files, dev_files = load_meta_data(c_dataset.datasets, eval_split=True, ignore_generated_eval=True)

wav_files = train_files + dev_files

# define Encoder model
model = setup_model(c)
model.load_state_dict(torch.load(args.model_path)["model"])
model.eval()
if args.use_cuda:
    model.cuda()

# compute speaker embeddings
speaker_mapping = {}
for idx, wav_file in enumerate(tqdm(wav_files)):
    if isinstance(wav_file, list):
        speaker_name = wav_file[2]
        wav_file = wav_file[1]
    else:
        speaker_name = None

    mel_spec = ap.melspectrogram(ap.load_wav(wav_file, sr=ap.sample_rate)).T
    mel_spec = torch.FloatTensor(mel_spec[None, :, :])
    if args.use_cuda:
        mel_spec = mel_spec.cuda()
    embedd = model.compute_embedding(mel_spec)
    embedd = embedd.detach().cpu().numpy()

    # create speaker_mapping if target dataset is defined
    wav_file_name = os.path.basename(wav_file)
    speaker_mapping[wav_file_name] = {}
    speaker_mapping[wav_file_name]["name"] = speaker_name
    speaker_mapping[wav_file_name]["embedding"] = embedd.flatten().tolist()

if speaker_mapping:
    # save speaker_mapping if target dataset is defined
    if '.json' not in args.output_path and '.npy' not in args.output_path:

        mapping_file_path = os.path.join(args.output_path, "speakers.json")
        mapping_npy_file_path = os.path.join(args.output_path, "speakers.npy")
    else:
        mapping_file_path = args.output_path.replace(".npy", ".json")
        mapping_npy_file_path = mapping_file_path.replace(".json", ".npy")

    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    if args.save_npy:
        np.save(mapping_npy_file_path, speaker_mapping)
        print("Speaker embeddings saved at:", mapping_npy_file_path)

    speaker_manager = SpeakerManager()
    # pylint: disable=W0212
    speaker_manager._save_json(mapping_file_path, speaker_mapping)
    print("Speaker embeddings saved at:", mapping_file_path)
