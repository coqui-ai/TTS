import os
import pickle as pickle_tts
from shutil import copyfile


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
