import json
import os
from typing import Dict, List

import fsspec
import numpy as np
import torch
from coqpit import Coqpit
from torch.utils.data.sampler import WeightedRandomSampler


class StyleManager:
    """Manage the styles for multi-style 🐸TTS models. Load a datafile and parse the information
    in a way that can be queried by style.

    Args:
        style_ids_file_path (str, optional): Path to the metafile that maps style names to ids used by
        TTS models. Defaults to "".
        config (Coqpit, optional): Coqpit config that contains the style information in the datasets filed.
        Defaults to None.

    Examples:
        >>> manager = StyleManager(style_ids_file_path=style_ids_file_path)
        >>> style_id_mapper = manager.style_ids
    """

    style_id_mapping: Dict = {}

    def __init__(
        self,
        style_ids_file_path: str = "",
        config: Coqpit = None,
    ):
        self.style_id_mapping = {}
        if style_ids_file_path:
            self.set_style_ids_from_file(style_ids_file_path)

        if config:
            self.set_style_ids_from_config(config)

    @staticmethod
    def _load_json(json_file_path: str) -> Dict:
        with fsspec.open(json_file_path, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_json(json_file_path: str, data: dict) -> None:
        with fsspec.open(json_file_path, "w") as f:
            json.dump(data, f, indent=4)

    @property
    def num_styles(self) -> int:
        return len(list(self.style_id_mapping.keys()))

    @property
    def style_names(self) -> List:
        return list(self.style_id_mapping.keys())

    @staticmethod
    def parse_style_ids_from_config(c: Coqpit) -> Dict:
        """Set style id from config.

        Args:
            c (Coqpit): Config

        Returns:
            Tuple[Dict, int]: style ID mapping and the number of styles.
        """
        styles = set({})
        for dataset in c.datasets:
            if "style" in dataset:
                styles.add(dataset["style"])
            else:
                raise ValueError(f"Dataset {dataset['name']} has no style specified.")
        return {name: i for i, name in enumerate(sorted(list(styles)))}

    def set_style_ids_from_config(self, c: Coqpit) -> None:
        """Set style IDs from config samples.

        Args:
            items (List): Data sampled returned by `load_meta_data()`.
        """
        self.style_id_mapping = self.parse_style_ids_from_config(c)

    def set_style_ids_from_file(self, file_path: str) -> None:
        """Load style ids from a json file.

        Args:
            file_path (str): Path to the target json file.
        """
        self.style_id_mapping = self._load_json(file_path)

    def save_style_ids_to_file(self, file_path: str) -> None:
        """Save style IDs to a json file.

        Args:
            file_path (str): Path to the output file.
        """
        self._save_json(file_path, self.style_id_mapping)


def _set_file_path(path):
    """Find the style_ids.json under the given path or the above it.
    Intended to band aid the different paths returned in restored and continued training."""
    path_restore = os.path.join(os.path.dirname(path), "style_ids.json")
    path_continue = os.path.join(path, "style_ids.json")
    fs = fsspec.get_mapper(path).fs
    if fs.exists(path_restore):
        return path_restore
    if fs.exists(path_continue):
        return path_continue
    return None


def get_style_weighted_sampler(items: list):
    style_names = np.array([item[3] for item in items])
    unique_style_names = np.unique(style_names).tolist()
    style_ids = [unique_style_names.index(l) for l in style_names]
    style_count = np.array([len(np.where(style_names == l)[0]) for l in unique_style_names])
    weight_style = 1.0 / style_count
    dataset_samples_weight = torch.from_numpy(np.array([weight_style[l] for l in style_ids])).double()
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))
