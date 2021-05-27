import argparse
import glob
import os

import torch
from tqdm import tqdm

from TTS.config import BaseDatasetConfig, load_config
from TTS.speaker_encoder.utils.generic_utils import setup_model
from TTS.tts.datasets import load_meta_data
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

parser = argparse.ArgumentParser(
    description='Compute embedding vectors for each wav file in a dataset. If "target_dataset" is defined, it generates "speakers.json" necessary for training a multi-speaker model.'
)
parser.add_argument("model_path", type=str, help="Path to model outputs (checkpoint, tensorboard etc.).")
parser.add_argument(
    "config_path",
    type=str,
    help="Path to config file for training.",
)
parser.add_argument("data_path", type=str, help="Data path for wav files - directory or CSV file")
parser.add_argument("output_path", type=str, help="path for output speakers.json.")
parser.add_argument(
    "--target_dataset",
    type=str,
    default="",
    help="Target dataset to pick a processor from TTS.tts.dataset.preprocess. Necessary to create a speakers.json file.",
)
parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
parser.add_argument("--separator", type=str, help="Separator used in file if CSV is passed for data_path", default="|")
args = parser.parse_args()


c = load_config(args.config_path)
ap = AudioProcessor(**c["audio"])

data_path = args.data_path
split_ext = os.path.splitext(data_path)
sep = args.separator

if args.target_dataset != "":
    # if target dataset is defined
    dataset_config = [
        BaseDatasetConfig(name=args.target_dataset, path=args.data_path, meta_file_train=None, meta_file_val=None),
    ]
    wav_files, _ = load_meta_data(dataset_config, eval_split=False)
else:
    # if target dataset is not defined
    if len(split_ext) > 0 and split_ext[1].lower() == ".csv":
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
                # print(f'wav_file: {wav_file}')
                if os.path.exists(wav_file):
                    wav_files.append(wav_file)
        print(f"Count of wavs imported: {len(wav_files)}")
    else:
        # Parse all wav files in data_path
        wav_files = glob.glob(data_path + "/**/*.wav", recursive=True)

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
    if ".json" not in args.output_path:
        mapping_file_path = os.path.join(args.output_path, "speakers.json")
    else:
        mapping_file_path = args.output_path
    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)
    speaker_manager = SpeakerManager()
    # pylint: disable=W0212
    speaker_manager._save_json(mapping_file_path, speaker_mapping)
    print("Speaker embeddings saved at:", mapping_file_path)
