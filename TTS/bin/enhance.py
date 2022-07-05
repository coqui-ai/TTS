import argparse
import glob
import os
from argparse import RawTextHelpFormatter
from distutils.dir_util import copy_tree

import torch
from librosa.core import load
from soundfile import write
from tqdm import tqdm

from TTS.enhancer.models import setup_model
from TTS.config import load_config

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="""Resample a folder recusively with librosa
                       Can be used in place or create a copy of the folder as an output.\n\n
                       Example run:
                            python TTS/bin/resample.py
                                --input_dir /root/LJSpeech-1.1/
                                --output_sr 22050
                                --output_dir /root/resampled_LJSpeech-1.1/
                                --file_ext wav
                                --n_jobs 24
                    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        required=True,
        help="Path of the checkpoint used for bandwith extension, it should contain a config.json file as well",
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        required=True,
        help="Path of the folder containing the audio files to resample",
    )

    parser.add_argument(
        "--file_ext",
        type=str,
        default="wav",
        required=False,
        help="Extension of the audio files to resample",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=False,
        help="Path of the destination folder. If not defined, the operation is done in place",
    )

    args = parser.parse_args()

    if args.output_dir:
        print("Recursively copying the input folder...")
        copy_tree(args.input_dir, args.output_dir)
        args.input_dir = args.output_dir

    print("Searching for the files...")
    audio_files = glob.glob(os.path.join(args.input_dir, f"**/*.{args.file_ext}"), recursive=True)
    print(f"Found {len(audio_files)} files...")

    print("Loading the upsampling model...")
    use_cuda = torch.cuda.is_available()
    config_path = os.path.join(os.path.split(args.model_path)[0], "config.json")
    config = load_config(config_path)
    model = setup_model(config)
    model.load_state_dict(torch.load(args.model_path)["model"])
    model.eval()
    if use_cuda:
        print("Using CUDA...")
        model.cuda()

    print("Upsampling the audio files...")
    with torch.no_grad():
        for wav_path in tqdm(audio_files):
            wav = load(wav_path, sr=config.input_sr)
            audio = torch.from_numpy(wav[0]).unsqueeze(0)
            if use_cuda:
                audio = audio.cuda()
            output = model.inference(audio)
            output = output.squeeze().cpu().detach().numpy()
            write(wav_path, output, config.target_sr)
