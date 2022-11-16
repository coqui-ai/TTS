import os
from typing import Any, Dict, List

import fsspec
import numpy as np
import torch
from coqpit import Coqpit

from TTS.config import check_config_and_model_args
from TTS.tts.utils.managers import BaseIDManager


class LanguageManager(BaseIDManager):
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

    def __init__(
        self,
        language_ids_file_path: str = "",
        config: Coqpit = None,
    ):
        super().__init__(id_file_path=language_ids_file_path)

        if config:
            self.set_language_ids_from_config(config)

    @property
    def num_languages(self) -> int:
        return len(list(self.name_to_id.keys()))

    @property
    def language_names(self) -> List:
        return list(self.name_to_id.keys())

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
            c (Coqpit): Config.
        """
        self.name_to_id = self.parse_language_ids_from_config(c)

    @staticmethod
    def parse_ids_from_data(items: List, parse_key: str) -> Any:
        raise NotImplementedError

    def set_ids_from_data(self, items: List, parse_key: str) -> Any:
        raise NotImplementedError

    def save_ids_to_file(self, file_path: str) -> None:
        """Save language IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.name_to_id)

    @staticmethod
    def init_from_config(config: Coqpit) -> "LanguageManager":
        """Initialize the language manager from a Coqpit config.

        Args:
            config (Coqpit): Coqpit config.
        """
        language_manager = None
        if check_config_and_model_args(config, "use_language_embedding", True):
            if config.get("language_ids_file", None):
                language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)
            language_manager = LanguageManager(config=config)
        return language_manager


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


def get_language_balancer_weights(items: list):
    language_names = np.array([item["language"] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1.0 / language_count
    # get weight for each sample
    dataset_samples_weight = np.array([weight_language[l] for l in language_ids])
    # normalize
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    return torch.from_numpy(dataset_samples_weight).float()
