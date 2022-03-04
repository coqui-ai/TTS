import argparse
import os
import torch
from argparse import RawTextHelpFormatter

from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager

parser = argparse.ArgumentParser(
    description="""Compute the accuracy of the encoder.\n\n"""
    """
    Example runs:
    python TTS/bin/eval_encoder.py emotion_encoder_model.pth.tar emotion_encoder_config.json  dataset_config.json
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
parser.add_argument("--use_cuda", type=bool, help="flag to set cuda.", default=True)
parser.add_argument("--eval", type=bool, help="compute eval.", default=True)

args = parser.parse_args()

c_dataset = load_config(args.config_dataset_path)

meta_data_train, meta_data_eval = load_tts_samples(c_dataset.datasets, eval_split=args.eval)
wav_files = meta_data_train + meta_data_eval

speaker_manager = SpeakerManager(
    encoder_model_path=args.model_path, encoder_config_path=args.config_path, use_cuda=args.use_cuda
)

if speaker_manager.speaker_encoder_config.map_classid_to_classname is not None:
    map_classid_to_classname = speaker_manager.speaker_encoder_config.map_classid_to_classname
else: 
    map_classid_to_classname = None

# compute speaker embeddings
class_acc_dict = {}

for idx, wav_file in enumerate(tqdm(wav_files)):
    if isinstance(wav_file, list):
        class_name = wav_file[2]
        wav_file = wav_file[1]
    else:
        class_name = None

    # extract the embedding
    embedd = speaker_manager.compute_d_vector_from_clip(wav_file)
    if speaker_manager.speaker_encoder_criterion is not None and map_classid_to_classname is not None:
        embedding = torch.FloatTensor(embedd).unsqueeze(0)
        if args.use_cuda:
            embedding = embedding.cuda()

        class_id = speaker_manager.speaker_encoder_criterion.softmax.inference(embedding).item()
        predicted_label = map_classid_to_classname[str(class_id)]
    else:
        predicted_label = None
    
    if class_name is not None and predicted_label is not None:
            is_equal = int(class_name == predicted_label)
            if class_name not in class_acc_dict:
                class_acc_dict[class_name] = [is_equal]
            else:
                class_acc_dict[class_name].append(is_equal)
    else:
        print("Error: class_name or/and predicted_label are None")
        exit()

acc_avg = 0
for key in class_acc_dict:
    acc = sum(class_acc_dict[key])/len(class_acc_dict[key])
    print("Class", key, "Accuracy:", acc)
    acc_avg += acc

print("Average Accuracy:", acc_avg/len(class_acc_dict))
