import sys
from collections import Counter
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Union

import numpy as np

from TTS.tts.datasets.dataset import *
from TTS.tts.datasets.formatters import *


def split_dataset(items):
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.

    Args:
        items (List[List]): A list of samples. Each sample is a list of `[audio_path, text, speaker_id]`.
    """
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


def load_tts_samples(
    datasets: Union[List[Dict], Dict], eval_split=True, formatter: Callable = None
) -> Tuple[List[List], List[List]]:
    """Parse the dataset from the datasets config, load the samples as a List and load the attention alignments if provided.
    If `formatter` is not None, apply the formatter to the samples else pick the formatter from the available ones based
    on the dataset name.

    Args:
        datasets (List[Dict], Dict): A list of datasets or a single dataset dictionary. If multiple datasets are
            in the list, they are all merged.

        eval_split (bool, optional): If true, create a evaluation split. If an eval split provided explicitly, generate
            an eval split automatically. Defaults to True.

        formatter (Callable, optional): The preprocessing function to be applied to create the list of samples. It
            must take the root_path and the meta_file name and return a list of samples in the format of
            `[[audio_path, text, speaker_id], ...]]`. See the available formatters in `TTS.tts.dataset.formatter` as
            example. Defaults to None.

    Returns:
        Tuple[List[List], List[List]: training and evaluation splits of the dataset.
    """
    meta_data_train_all = []
    meta_data_eval_all = [] if eval_split else None
    if not isinstance(datasets, list):
        datasets = [datasets]
    for dataset in datasets:
        name = dataset["name"]
        root_path = dataset["path"]
        meta_file_train = dataset["meta_file_train"]
        meta_file_val = dataset["meta_file_val"]
        ignored_speakers = dataset["ignored_speakers"]
        language = dataset["language"]

        # setup the right data processor
        if formatter is None:
            formatter = _get_formatter_by_name(name)
        # load train set
        meta_data_train = formatter(root_path, meta_file_train, ignored_speakers=ignored_speakers)
        meta_data_train = [[*item, language] for item in meta_data_train]

        print(f" | > Found {len(meta_data_train)} files in {Path(root_path).resolve()}")
        # load evaluation split if set
        if eval_split:
            if meta_file_val:
                meta_data_eval = formatter(root_path, meta_file_val, ignored_speakers=ignored_speakers)
                meta_data_eval = [[*item, language] for item in meta_data_eval]
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
        # set none for the next iter
        formatter = None
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


def _get_formatter_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def find_unique_chars(data_samples, verbose=True):
    texts = "".join(item[0] for item in data_samples)
    chars = set(texts)
    lower_chars = filter(lambda c: c.islower(), chars)
    chars_force_lower = [c.lower() for c in chars]
    chars_force_lower = set(chars_force_lower)

    if verbose:
        print(f" > Number of unique characters: {len(chars)}")
        print(f" > Unique characters: {''.join(sorted(chars))}")
        print(f" > Unique lower characters: {''.join(sorted(lower_chars))}")
        print(f" > Unique all forced to lower characters: {''.join(sorted(chars_force_lower))}")
    return chars_force_lower
