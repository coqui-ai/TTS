import os
import json
import torch
import fsspec
import numpy as np
from typing import Dict, Tuple, List
from coqpit import Coqpit

from torch.utils.data.sampler import WeightedRandomSampler

class LanguageManager:
    """Manage the languages for multi-lingual 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by language.

    Args:
        language_id_file_path (str, optional): Path to the metafile that maps language names to ids used by
        TTS models. Defaults to "".

    Examples:
        >>> manager = LanguageManager(language_id_file_path=language_id_file_path)
        >>> language_id_mapper = manager.language_ids
    """
    language_id_mapping: Dict = {}
    def __init__(
        self,
        language_id_file_path: str = "",
    ):
        if language_id_file_path:
            self.set_language_ids_from_file(language_id_file_path)

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
    def parse_languages_from_data(items: list) -> Tuple[Dict, int]:
        """Parse language IDs from data samples retured by `load_meta_data()`.

        Args:
            items (list): Data sampled returned by `load_meta_data()`.

        Returns:
            Tuple[Dict, int]: language IDs and number of languages.
        """
        languages = sorted({item[3] for item in items})
        language_ids = {name: i for i, name in enumerate(languages)}
        num_languages = len(language_ids)
        return language_ids, num_languages

    def set_language_ids_from_data(self, items: List) -> None:
        """Set language IDs from data samples.

        Args:
            items (List): Data sampled returned by `load_meta_data()`.
        """
        self.language_id_mapping, _ = self.parse_languages_from_data(items)

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

def get_language_manager(c: Coqpit, data: List = None, restore_path: str = None) -> LanguageManager:
    """Initiate a `LanguageManager` instance by the provided config.

    Args:
        c (Coqpit): Model configuration.
        restore_path (str): Path to a previous training folder.
        data (List): Data sampled returned by `load_meta_data()`. Defaults to None.
        out_path (str, optional): Save the generated language IDs to a output path. Defaults to None.

    Returns:
        SpeakerManager: initialized and ready to use instance.
    """
    language_manager = LanguageManager()
    if c.use_language_embedding:
        if data is not None:
            language_manager.set_language_ids_from_data(data)
        if restore_path:
            language_file = _set_file_path(restore_path)
            # restoring language manager from a previous run.
            if language_file:
                language_manager.set_language_ids_from_file(language_file)
        if  language_manager.num_languages > 0:
            print(
                " > Language manager is loaded with {} languages: {}".format(
                    language_manager.num_languages, ", ".join(language_manager.language_names)
                )
            )
    return language_manager

def get_language_weighted_sampler(items: list):
    language_names = np.array([item[3] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1. / language_count
    dataset_samples_weight = torch.from_numpy(np.array([weight_language[l] for l in language_ids])).double()
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))
