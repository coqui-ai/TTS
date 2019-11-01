import argparse
import glob
import os

import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from TTS.datasets.preprocess import get_preprocessor_by_name
from TTS.speaker_encoder.dataset import MyDataset
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.speaker_encoder.visual import plot_embeddings
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import load_config

parser = argparse.ArgumentParser(
    description='Compute embedding vectors for each wav file in a dataset. ')
parser.add_argument(
    'model_path',
    type=str,
    help='Path to model outputs (checkpoint, tensorboard etc.).')
parser.add_argument(
    'config_path',
    type=str,
    help='Path to config file for training.',
)
parser.add_argument(
    'data_path',
    type=str,
    help='Defines the data path. It overwrites config.json.')
parser.add_argument(
    'output_path',
    type=str,
    help='path for training outputs.')
parser.add_argument(
    '--use_cuda', type=bool, help='flag to set cuda.', default=False
)
args = parser.parse_args()


c = load_config(args.config_path)
ap = AudioProcessor(**c['audio'])

wav_files = glob.glob(args.data_path + '/**/*.wav', recursive=True)
output_files = [wav_file.replace(args.data_path, args.output_path).replace(
    '.wav', '.npy') for wav_file in wav_files]

for output_file in output_files:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

model = SpeakerEncoder(**c.model)
model.load_state_dict(torch.load(args.model_path)['model'])
model.eval()
if args.use_cuda:
    model.cuda()

for idx, wav_file in enumerate(tqdm(wav_files)):
    mel_spec = ap.melspectrogram(ap.load_wav(wav_file)).T
    mel_spec = torch.FloatTensor(mel_spec[None, :, :])
    if args.use_cuda:
        mel_spec = mel_spec.cuda()
    embedd = model.compute_embedding(mel_spec)
    np.save(output_files[idx], embedd.detach().cpu().numpy())
