import argparse
import os
from argparse import RawTextHelpFormatter

import torch
from tqdm import tqdm

from TTS.config import load_config
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.speakers import SpeakerManager

parser = argparse.ArgumentParser(
    description="""Compute embedding vectors for each audio file in a dataset and store them keyed by `{dataset_name}#{file_path}` in a .pth file\n\n"""
    """
    Example runs:
    python TTS/bin/compute_embeddings.py --model_path speaker_encoder_model.pth --config_path speaker_encoder_config.json  --config_dataset_path dataset_config.json

    python TTS/bin/compute_embeddings.py --model_path speaker_encoder_model.pth --config_path speaker_encoder_config.json  --fomatter vctk --dataset_path /path/to/vctk/dataset --dataset_name my_vctk --metafile /path/to/vctk/metafile.csv
    """,
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument(
    "--model_path",
    type=str,
    help="Path to model checkpoint file. It defaults to the released speaker encoder.",
    default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar",
)
parser.add_argument(
    "--config_path",
    type=str,
    help="Path to model config file. It defaults to the released speaker encoder config.",
    default="https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json",
)
parser.add_argument(
    "--config_dataset_path",
    type=str,
    help="Path to dataset config file. You either need to provide this or `formatter_name`, `dataset_name` and `dataset_path` arguments.",
    default=None,
)
parser.add_argument("--output_path", type=str, help="Path for output `pth` or `json` file.", default="speakers.pth")
parser.add_argument("--old_file", type=str, help="Previous embedding file to only compute new audios.", default=None)
parser.add_argument("--disable_cuda", type=bool, help="Flag to disable cuda.", default=False)
parser.add_argument("--no_eval", type=bool, help="Do not compute eval?. Default False", default=False)
parser.add_argument(
    "--formatter_name",
    type=str,
    help="Name of the formatter to use. You either need to provide this or `config_dataset_path`",
    default=None,
)
parser.add_argument(
    "--dataset_name",
    type=str,
    help="Name of the dataset to use. You either need to provide this or `config_dataset_path`",
    default=None,
)
parser.add_argument(
    "--dataset_path",
    type=str,
    help="Path to the dataset. You either need to provide this or `config_dataset_path`",
    default=None,
)
parser.add_argument(
    "--metafile",
    type=str,
    help="Path to the meta file. If not set, dataset formatter uses the default metafile if it is defined in the formatter. You either need to provide this or `config_dataset_path`",
    default=None,
)
args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.disable_cuda

if args.config_dataset_path is not None:
    c_dataset = load_config(args.config_dataset_path)
    meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not args.no_eval)
else:
    c_dataset = BaseDatasetConfig()
    c_dataset.formatter = args.formatter_name
    c_dataset.dataset_name = args.dataset_name
    c_dataset.path = args.dataset_path
    c_dataset.meta_file_train = args.metafile if args.metafile else None
    meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=not args.no_eval)


if meta_data_eval is None:
    samples = meta_data_train
else:
    samples = meta_data_train + meta_data_eval

encoder_manager = SpeakerManager(
    encoder_model_path=args.model_path,
    encoder_config_path=args.config_path,
    d_vectors_file_path=args.old_file,
    use_cuda=use_cuda,
)

class_name_key = encoder_manager.encoder_config.class_name_key

# compute speaker embeddings
speaker_mapping = {}
for idx, fields in enumerate(tqdm(samples)):
    class_name = fields[class_name_key]
    audio_file = fields["audio_file"]
    embedding_key = fields["audio_unique_name"]
    root_path = fields["root_path"]

    if args.old_file is not None and embedding_key in encoder_manager.clip_ids:
        # get the embedding from the old file
        embedd = encoder_manager.get_embedding_by_clip(embedding_key)
    else:
        # extract the embedding
        embedd = encoder_manager.compute_embedding_from_clip(audio_file)

    # create speaker_mapping if target dataset is defined
    speaker_mapping[embedding_key] = {}
    speaker_mapping[embedding_key]["name"] = class_name
    speaker_mapping[embedding_key]["embedding"] = embedd

if speaker_mapping:
    # save speaker_mapping if target dataset is defined
    if os.path.isdir(args.output_path):
        mapping_file_path = os.path.join(args.output_path, "speakers.pth")
    else:
        mapping_file_path = args.output_path

    if os.path.dirname(mapping_file_path) != "":
        os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    save_file(speaker_mapping, mapping_file_path)
    print("Speaker embeddings saved at:", mapping_file_path)
