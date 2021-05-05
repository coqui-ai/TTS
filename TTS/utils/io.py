import json
import os
import pickle as pickle_tts
import re
from shutil import copyfile

import yaml

from TTS.utils.generic_utils import find_module

from .generic_utils import find_module


class RenamingUnpickler(pickle_tts.Unpickler):
    """Overload default pickler to solve module renaming problem"""

    def find_class(self, module, name):
        return super().find_class(module.replace("mozilla_voice_tts", "TTS"), name)


class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__dict__ = self


def read_json_with_comments(json_path):
    """DEPRECATED"""
    # fallback to json
    with open(json_path, "r", encoding="utf-8") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r"\\\n", "", input_str)
    input_str = re.sub(r"//.*\n", "\n", input_str)
    data = json.loads(input_str)
    return data


def load_config(config_path: str) -> None:
    config_dict = {}
    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif ext == ".json":
        with open(config_path, "r", encoding="utf-8") as f:
            input_str = f.read()
        data = json.loads(input_str)
    else:
        raise TypeError(f" [!] Unknown config file type {ext}")
    config_dict.update(data)
    config_class = find_module("TTS.tts.configs", config_dict["model"].lower() + "_config")
    config = config_class()
    config.from_dict(config_dict)
    return config


def copy_model_files(config, out_path, new_fields):
    """Copy config.json and other model files to training folder and add
    new fields.

    Args:
        config (Coqpit): Coqpit config defining the training run.
        out_path (str): output path to copy the file.
        new_fields (dict): new fileds to be added or edited
            in the config file.
    """
    copy_config_path = os.path.join(out_path, "config.json")
    # add extra information fields
    config.update(new_fields, allow_new=True)
    config.save_json(copy_config_path)
    # copy model stats file if available
    if config.audio.stats_path is not None:
        copy_stats_path = os.path.join(out_path, "scale_stats.npy")
        if not os.path.exists(copy_stats_path):
            copyfile(
                config.audio.stats_path,
                copy_stats_path,
            )
