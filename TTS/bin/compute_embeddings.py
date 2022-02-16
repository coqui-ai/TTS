import argparse
import os
from argparse import RawTextHelpFormatter

from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

parser = argparse.ArgumentParser(
    description="""Compute embedding vectors for each wav file in a dataset.\n\n"""
    """
    Example runs:
    python TTS/bin/compute_embeddings.py speaker_encoder_model.pth.tar speaker_encoder_config.json  dataset_config.json embeddings_output_path/
    """,
    formatter_class=RawTextHelpFormatter,
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
parser.add_argument(
    "--old_file", type=str, help="Previous speakers.json file, only compute for new audios.", default=None
)
parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
parser.add_argument("--eval", type=bool, help="compute eval.", default=True)

args = parser.parse_args()

c_dataset = load_config(args.config_dataset_path)

meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=args.eval)
wav_files = meta_data_train + meta_data_eval

speaker_manager = SpeakerManager(
    encoder_model_path=args.model_path,
    encoder_config_path=args.config_path,
    d_vectors_file_path=args.old_file,
    use_cuda=args.use_cuda,
)

# compute speaker embeddings
speaker_mapping = {}
for idx, wav_file in enumerate(tqdm(wav_files)):
    if isinstance(wav_file, list):
        speaker_name = wav_file[2]
        wav_file = wav_file[1]
    else:
        speaker_name = None

    wav_file_name = os.path.basename(wav_file)
    if args.old_file is not None and wav_file_name in speaker_manager.clip_ids:
        # get the embedding from the old file
        embedd = speaker_manager.get_d_vector_by_clip(wav_file_name)
    else:
        # extract the embedding
        embedd = speaker_manager.compute_d_vector_from_clip(wav_file)

    # create speaker_mapping if target dataset is defined
    speaker_mapping[wav_file_name] = {}
    speaker_mapping[wav_file_name]["name"] = speaker_name
    speaker_mapping[wav_file_name]["embedding"] = embedd

if speaker_mapping:
    # save speaker_mapping if target dataset is defined
    if ".json" not in args.output_path:
        mapping_file_path = os.path.join(args.output_path, "speakers.json")
    else:
        mapping_file_path = args.output_path

    os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    # pylint: disable=W0212
    speaker_manager._save_json(mapping_file_path, speaker_mapping)
    print("Speaker embeddings saved at:", mapping_file_path)
