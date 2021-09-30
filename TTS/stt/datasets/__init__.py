import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np

from TTS.stt.datasets.formatters import *


def split_dataset(items, eval_split_size=None):
    if not eval_split_size:
        eval_split_size = min(500, int(len(items) * 0.01))
    assert eval_split_size > 0, " [!] You do not have enough samples to train. You need at least 100 samples."
    np.random.seed(0)
    np.random.shuffle(items)
    return items[:eval_split_size], items[eval_split_size:]


def load_stt_samples(
    datasets: Union[Dict, List[Dict]], eval_split=True, eval_split_size: int = None
) -> Tuple[List[List], List[List]]:
    """Parse the dataset, load the samples as a list and load the attention alignments if provided.

    Args:
        datasets (Union[Dict, List[Dict]]): A list of dataset configs or a single dataset config.
        eval_split (bool, optional): If true, create a evaluation split. If an eval split provided explicitly, generate
            an eval split automatically. Defaults to True.
        eval_split_size (int, optional): The size of the evaluation split. If None, use pre-defined split size.
            Defaults to None.

    Returns:
        Tuple[List[List], List[List]: training and evaluation splits of the dataset.
    """
    if not isinstance(datasets, list):
        datasets = [datasets]
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
        # load evaluation split if set
        if eval_split:
            if meta_file_val:
                meta_data_eval = preprocessor(root_path, meta_file_val)
            else:
                meta_data_eval, meta_data_train = split_dataset(meta_data_train, eval_split_size)
            meta_data_eval_all += meta_data_eval
        meta_data_train_all += meta_data_train
    return meta_data_train_all, meta_data_eval_all


def _get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())
