import argparse
import os
from argparse import RawTextHelpFormatter

import torch
from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.managers import save_file
from TTS.tts.utils.managers import EmbeddingManager

parser = argparse.ArgumentParser(
    description="""Compute embedding vectors for each wav file in a dataset.\n\n"""
    """
    Example runs:
    python TTS/bin/compute_embeddings.py speaker_encoder_model.pth speaker_encoder_config.json  dataset_config.json
    """,
    formatter_class=RawTextHelpFormatter,
)
parser.add_argument("model_path", type=str, help="Path to model checkpoint file.")
parser.add_argument("config_path", type=str, help="Path to model config file.")
parser.add_argument("config_dataset_path", type=str, help="Path to dataset config file.")
parser.add_argument("--output_path", type=str, help="Path for output `pth` or `json` file.", default="speakers.pth")
parser.add_argument("--old_file", type=str, help="Previous embedding file to only compute new audios.", default=None)
parser.add_argument("--disable_cuda", type=bool, help="Flag to disable cuda.", default=False)
parser.add_argument("--no_eval", type=bool, help="Do not compute eval?. Default False", default=False)
parser.add_argument("--use_predicted_label", type=bool, help="If True and predicted label is available with will use it.", default=False)

args = parser.parse_args()

use_cuda = torch.cuda.is_available() and not args.disable_cuda

c_dataset = load_config(args.config_dataset_path)

meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=not args.no_eval)

if meta_data_eval is None:
    wav_files = meta_data_train
else:
    wav_files = meta_data_train + meta_data_eval

encoder_manager = EmbeddingManager(
    encoder_model_path=args.model_path,
    encoder_config_path=args.config_path,
    embedding_file_path=args.old_file,
    use_cuda=use_cuda,
)

class_name_key = encoder_manager.encoder_config.class_name_key

# compute speaker embeddings
class_mapping = {}
for idx, wav_file in enumerate(tqdm(wav_files)):
    if isinstance(wav_file, dict):
        class_name = wav_file[class_name_key] if class_name_key in wav_file else None
        wav_file = wav_file["audio_file"]
    else:
        class_name = None

    wav_file_name = os.path.basename(wav_file)
    if args.old_file is not None and wav_file_name in encoder_manager.clip_ids:
        # get the embedding from the old file
        embedd = encoder_manager.get_embedding_by_clip(wav_file_name)
    else:
        # extract the embedding
        embedd = encoder_manager.compute_embedding_from_clip(wav_file)

    if args.use_predicted_label:
        map_classid_to_classname = getattr(encoder_manager.encoder_config, "map_classid_to_classname", None)
        if encoder_manager.encoder_criterion is not None and map_classid_to_classname is not None:
            embedding = torch.FloatTensor(embedd).unsqueeze(0)
            if encoder_manager.use_cuda:
                embedding = embedding.cuda()

            class_id = encoder_manager.encoder_criterion.softmax.inference(embedding).item()
            class_name = map_classid_to_classname[str(class_id)]
        else:
            raise RuntimeError(" [!] use_predicted_label is enable and predicted_labels is not available !!")

    # create class_mapping if target dataset is defined
    class_mapping[wav_file_name] = {}
    class_mapping[wav_file_name]["name"] = class_name
    class_mapping[wav_file_name]["embedding"] = embedd

if args.old_file:
    # merge the embeddings dict
    class_mapping = {**encoder_manager.embeddings, **class_mapping}

if class_mapping:
    # save class_mapping if target dataset is defined
    if ".json" not in args.output_path or ".pth" not in args.output_path:
        if class_name_key == "speaker_name":
            mapping_file_path = os.path.join(args.output_path, "speakers.pth")
        else:
            mapping_file_path = os.path.join(args.output_path, "emotions.pth")
    else:
        mapping_file_path = args.output_path

    if os.path.dirname(mapping_file_path) != "":
        os.makedirs(os.path.dirname(mapping_file_path), exist_ok=True)

    save_file(class_mapping, mapping_file_path)
    print("Embeddings saved at:", mapping_file_path)
