import json
import os
from typing import Dict, List

import fsspec
import numpy as np
import torch
from coqpit import Coqpit
from torch.utils.data.sampler import WeightedRandomSampler


class LanguageManager:
    """Manage the languages for multi-lingual 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by language.

    Args:
        language_ids_file_path (str, optional): Path to the metafile that maps language names to ids used by
        TTS models. Defaults to "".
        config (Coqpit, optional): Coqpit config that contains the language information in the datasets filed.
        Defaults to None.

    Examples:
        >>> manager = LanguageManager(language_ids_file_path=language_ids_file_path)
        >>> language_id_mapper = manager.language_ids
    """

    language_id_mapping: Dict = {}

    def __init__(
        self,
        language_ids_file_path: str = "",
        config: Coqpit = None,
    ):
        self.language_id_mapping = {}
        if language_ids_file_path:
            self.set_language_ids_from_file(language_ids_file_path)

        if config:
            self.set_language_ids_from_config(config)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with fsspec.open(json_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with fsspec.open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_languages(self) -> int:
        return len(list(self.language_id_mapping.keys()))

    @property
    def language_names(self) -> List:
        return list(self.language_id_mapping.keys())

    @staticmethod
    def parse_language_ids_from_config(c: Coqpit) -> Dict:
        """Set language id from config.

        Args:
            c (Coqpit): Config

        Returns:
            Tuple[Dict, int]: Language ID mapping and the number of languages.
        """
        languages = set({})
        for dataset in c.datasets:
            if "language" in dataset:
                languages.add(dataset["language"])
            else:
                raise ValueError(f"Dataset {dataset['name']} has no language specified.")
        return {name: i for i, name in enumerate(sorted(list(languages)))}

    def set_language_ids_from_config(self, c: Coqpit) -> None:
        """Set language IDs from config samples.

        Args:
            items (List): Data sampled returned by `load_meta_data()`.
        """
        self.language_id_mapping = self.parse_language_ids_from_config(c)

    def set_language_ids_from_file(self, file_path: str) -> None:
        """Load language ids from a json file.

        Args:
            file_path (str): Path to the target json file.
        """
        self.language_id_mapping = self._load_json(file_path)

    def save_language_ids_to_file(self, file_path: str) -> None:
        """Save language IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.language_id_mapping)


def _set_file_path(path):
    """Find the language_ids.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "language_ids.json")
    path_continue = os.path.join(path, "language_ids.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    return None


def get_language_weighted_sampler(items: list):
    language_names = np.array([item[3] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1.0 / language_count
    dataset_samples_weight = torch.from_numpy(np.array([weight_language[l] for l in language_ids])).double()
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))
