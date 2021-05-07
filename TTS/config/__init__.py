import json
import os

import yaml

from TTS.config.shared_configs import *
from TTS.utils.generic_utils import find_module


def _search_configs(model_name):
    config_class = None
    paths = ["TTS.tts.configs", "TTS.vocoder.configs", "TTS.speaker_encoder"]
    for path in paths:
        try:
            config_class = find_module(path, model_name + "_config")
        except ModuleNotFoundError:
            pass
    if config_class is None:
        raise ModuleNotFoundError()
    return config_class


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
    config_class = _search_configs(config_dict["model"].lower())
    config = config_class()
    config.from_dict(config_dict)
    return config
