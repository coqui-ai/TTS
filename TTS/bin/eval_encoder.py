import argparse
from argparse import RawTextHelpFormatter

import torch
from tqdm import tqdm

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.speakers import SpeakerManager


def compute_encoder_accuracy(dataset_items, encoder_manager):
    class_name_key = encoder_manager.encoder_config.class_name_key
    map_classid_to_classname = getattr(encoder_manager.encoder_config, "map_classid_to_classname", None)

    class_acc_dict = {}

    # compute embeddings for all wav_files
    for item in tqdm(dataset_items):
        class_name = item[class_name_key]
        wav_file = item["audio_file"]

        # extract the embedding
        embedd = encoder_manager.compute_embedding_from_clip(wav_file)
        if encoder_manager.encoder_criterion is not None and map_classid_to_classname is not None:
            embedding = torch.FloatTensor(embedd).unsqueeze(0)
            if encoder_manager.use_cuda:
                embedding = embedding.cuda()

            class_id = encoder_manager.encoder_criterion.softmax.inference(embedding).item()
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
            raise RuntimeError("Error: class_name or/and predicted_label are None")

    acc_avg = 0
    for key, values in class_acc_dict.items():
        acc = sum(values) / len(values)
        print("Class", key, "Accuracy:", acc)
        acc_avg += acc

    print("Average Accuracy:", acc_avg / len(class_acc_dict))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="""Compute the accuracy of the encoder.\n\n"""
        """
        Example runs:
        python TTS/bin/eval_encoder.py emotion_encoder_model.pth emotion_encoder_config.json  dataset_config.json
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
    items = meta_data_train + meta_data_eval

    enc_manager = SpeakerManager(
        encoder_model_path=args.model_path, encoder_config_path=args.config_path, use_cuda=args.use_cuda
    )

    compute_encoder_accuracy(items, enc_manager)
