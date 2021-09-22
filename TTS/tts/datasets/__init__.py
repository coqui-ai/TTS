import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from TTS.tts.datasets.formatters import *
from TTS.tts.datasets.TTSDataset import TTSDataset


def split_dataset(items):
    speakers = [item[-1] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = min(500, int(len(items) * 0.01))
    assert eval_split_size > 0, " [!] You do not have enough samples to train. You need at least 100 samples."
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item[-1] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx][-1]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]


def load_meta_data(datasets: List[Dict], eval_split=True) -> Tuple[List[List], List[List]]:
    """Parse the dataset, load the samples as a list and load the attention alignments if provided.

    Args:
        datasets (List[Dict]): A list of dataset dictionaries or dataset configs.
        eval_split (bool, optional): If true, create a evaluation split. If an eval split provided explicitly, generate
            an eval split automatically. Defaults to True.

    Returns:
        Tuple[List[List], List[List]: training and evaluation splits of the dataset.
    """
    meta_data_train_all = []
    meta_data_eval_all = [] if eval_split else None
    for dataset in datasets:
        name = dataset["name"]
        root_path = dataset["path"]
        meta_file_train = dataset["meta_file_train"]
        meta_file_val = dataset["meta_file_val"]
        # setup the right data processor
        preprocessor = _get_preprocessor_by_name(name)
        # load train set
        meta_data_train = preprocessor(root_path, meta_file_train)
        print(f" | > Found {len(meta_data_train)} files in {Path(root_path).resolve()}")
        # load evaluation split if set
        if eval_split:
            if meta_file_val:
                meta_data_eval = preprocessor(root_path, meta_file_val)
            else:
                meta_data_eval, meta_data_train = split_dataset(meta_data_train)
            meta_data_eval_all += meta_data_eval
        meta_data_train_all += meta_data_train
        # load attention masks for the duration predictor training
        if dataset.meta_file_attn_mask:
            meta_data = dict(load_attention_mask_meta_data(dataset["meta_file_attn_mask"]))
            for idx, ins in enumerate(meta_data_train_all):
                attn_file = meta_data[ins[1]].strip()
                meta_data_train_all[idx].append(attn_file)
            if meta_data_eval_all:
                for idx, ins in enumerate(meta_data_eval_all):
                    attn_file = meta_data[ins[1]].strip()
                    meta_data_eval_all[idx].append(attn_file)
    return meta_data_train_all, meta_data_eval_all


def load_attention_mask_meta_data(metafile_path):
    """Load meta data file created by compute_attention_masks.py"""
    with open(metafile_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    meta_data = []
    for line in lines:
        wav_file, attn_file = line.split("|")
        meta_data.append([wav_file, attn_file])
    return meta_data


def _get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())
