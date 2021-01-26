import os
import re
import json
import yaml
import pickle as pickle_tts
from shutil import copyfile


class RenamingUnpickler(pickle_tts.Unpickler):
    """Overload default pickler to solve module renaming problem"""
    def find_class(self, module, name):
        return super().find_class(module.replace('mozilla_voice_tts', 'TTS'), name)


class AttrDict(dict):
    """A custom dict which converts dict keys
    to class attributes"""
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def read_json_with_comments(json_path):
    # fallback to json
    with open(json_path, "r") as f:
        input_str = f.read()
    # handle comments
    input_str = re.sub(r'\\\n', '', input_str)
    input_str = re.sub(r'//.*\n', '\n', input_str)
    data = json.loads(input_str)
    return data

def load_config(config_path: str) -> AttrDict:
    """Load config files and discard comments

    Args:
        config_path (str): path to config file.
    """
    config = AttrDict()

    ext = os.path.splitext(config_path)[1]
    if ext in (".yml", ".yaml"):
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    else:
        data = read_json_with_comments(config_path)
    config.update(data)
    return config


def copy_model_files(c, config_file, out_path, new_fields):
    """Copy config.json and other model files to training folder and add
    new fields.

    Args:
        c (dict): model config from config.json.
        config_file (str): path to config file.
        out_path (str): output path to copy the file.
        new_fields (dict): new fileds to be added or edited
            in the config file.
    """
    # copy config.json
    copy_config_path = os.path.join(out_path, 'config.json')
    config_lines = open(config_file, "r").readlines()
    # add extra information fields
    for key, value in new_fields.items():
        if isinstance(value, str):
            new_line = '"{}":"{}",\n'.format(key, value)
        else:
            new_line = '"{}":{},\n'.format(key, value)
        config_lines.insert(1, new_line)
    config_out_file = open(copy_config_path, "w")
    config_out_file.writelines(config_lines)
    config_out_file.close()
    # copy model stats file if available
    if c.audio['stats_path'] is not None:
        copy_stats_path = os.path.join(out_path, 'scale_stats.npy')
        copyfile(c.audio['stats_path'], copy_stats_path)
