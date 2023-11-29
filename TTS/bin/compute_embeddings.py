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


def compute_embeddings(
    model_path,
    config_path,
    output_path,
    old_speakers_file=None,
    old_append=False,
    config_dataset_path=None,
    formatter_name=None,
    dataset_name=None,
    dataset_path=None,
    meta_file_train=None,
    meta_file_val=None,
    disable_cuda=False,
    no_eval=False,
):
    use_cuda = torch.cuda.is_available() and not disable_cuda

    if config_dataset_path is not None:
        c_dataset = load_config(config_dataset_path)
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not no_eval)
    else:
        c_dataset = BaseDatasetConfig()
        c_dataset.formatter = formatter_name
        c_dataset.dataset_name = dataset_name
        c_dataset.path = dataset_path
        if meta_file_train is not None:
            c_dataset.meta_file_train = meta_file_train
        if meta_file_val is not None:
            c_dataset.meta_file_val = meta_file_val
        meta_data_train, meta_data_eval = load_tts_samples(c_dataset, eval_split=not no_eval)

    if meta_data_eval is None:
        samples = meta_data_train
    else:
        samples = meta_data_train + meta_data_eval

    encoder_manager = SpeakerManager(
        encoder_model_path=model_path,
        encoder_config_path=config_path,
        d_vectors_file_path=old_speakers_file,
        use_cuda=use_cuda,
    )

    class_name_key = encoder_manager.encoder_config.class_name_key

    # compute speaker embeddings
    if old_speakers_file is not None and old_append:
        speaker_mapping = encoder_manager.embeddings
    else:
        speaker_mapping = {}

    for fields in tqdm(samples):
        class_name = fields[class_name_key]
        audio_file = fields["audio_file"]
        embedding_key = fields["audio_unique_name"]

        # Only update the speaker name when the embedding is already in the old file.
        if embedding_key in speaker_mapping:
            speaker_mapping[embedding_key]["name"] = class_name
            continue

        if old_speakers_file is not None and embedding_key in encoder_manager.clip_ids:
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
        if os.path.isdir(output_path):
            mapping_file_path = os.path.join(output_path, "speakers.pth")
        else:
            mapping_file_path = output_path

        if os.path.dirname(mapping_file_path) != "":
            os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

        save_file(speaker_mapping, mapping_file_path)
        print("Speaker embeddings saved at:", mapping_file_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute embedding vectors for each audio file in a dataset and store them keyed by `{dataset_name}#{file_path}` in a .pth file\n\n"""
        """
        Example runs:
        python TTS/bin/compute_embeddings.py --model_path speaker_encoder_model.pth --config_path speaker_encoder_config.json  --config_dataset_path dataset_config.json

        python TTS/bin/compute_embeddings.py --model_path speaker_encoder_model.pth --config_path speaker_encoder_config.json  --formatter_name coqui --dataset_path /path/to/vctk/dataset --dataset_name my_vctk --meta_file_train /path/to/vctk/metafile_train.csv --meta_file_val /path/to/vctk/metafile_eval.csv
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
    parser.add_argument(
        "--output_path",
        type=str,
        help="Path for output `pth` or `json` file.",
        default="speakers.pth",
    )
    parser.add_argument(
        "--old_file",
        type=str,
        help="The old existing embedding file, from which the embeddings will be directly loaded for already computed audio clips.",
        default=None,
    )
    parser.add_argument(
        "--old_append",
        help="Append new audio clip embeddings to the old embedding file, generate a new non-duplicated merged embedding file. Default False",
        default=False,
        action="store_true",
    )
    parser.add_argument("--disable_cuda", type=bool, help="Flag to disable cuda.", default=False)
    parser.add_argument("--no_eval", help="Do not compute eval?. Default False", default=False, action="store_true")
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
        "--meta_file_train",
        type=str,
        help="Path to the train meta file. If not set, dataset formatter uses the default metafile if it is defined in the formatter. You either need to provide this or `config_dataset_path`",
        default=None,
    )
    parser.add_argument(
        "--meta_file_val",
        type=str,
        help="Path to the evaluation meta file. If not set, dataset formatter uses the default metafile if it is defined in the formatter. You either need to provide this or `config_dataset_path`",
        default=None,
    )
    args = parser.parse_args()

    compute_embeddings(
        args.model_path,
        args.config_path,
        args.output_path,
        old_speakers_file=args.old_file,
        old_append=args.old_append,
        config_dataset_path=args.config_dataset_path,
        formatter_name=args.formatter_name,
        dataset_name=args.dataset_name,
        dataset_path=args.dataset_path,
        meta_file_train=args.meta_file_train,
        meta_file_val=args.meta_file_val,
        disable_cuda=args.disable_cuda,
        no_eval=args.no_eval,
    )
