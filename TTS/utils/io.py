import json
import os
import pickle as pickle_tts
import shutil
from typing import Any, Callable, Dict, Union

import fsspec
import torch
from coqpit import Coqpit

from TTS.utils.generic_utils import get_user_data_dir


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


def copy_model_files(config: Coqpit, out_path, new_fields=None):
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
    if new_fields:
        config.update(new_fields, allow_new=True)
    # TODO: Revert to config.save_json() once Coqpit supports arbitrary paths.
    with fsspec.open(copy_config_path, "w", encoding="utf8") as f:
        json.dump(config.to_dict(), f, indent=4)

    # copy model stats file if available
    if config.audio.stats_path is not None:
        copy_stats_path = os.path.join(out_path, "scale_stats.npy")
        filesystem = fsspec.get_mapper(copy_stats_path).fs
        if not filesystem.exists(copy_stats_path):
            with fsspec.open(config.audio.stats_path, "rb") as source_file:
                with fsspec.open(copy_stats_path, "wb") as target_file:
                    shutil.copyfileobj(source_file, target_file)


def load_fsspec(
    path: str,
    map_location: Union[str, Callable, torch.device, Dict[Union[str, torch.device], Union[str, torch.device]]] = None,
    cache: bool = True,
    **kwargs,
) -> Any:
    """Like torch.load but can load from other locations (e.g. s3:// , gs://).

    Args:
        path: Any path or url supported by fsspec.
        map_location: torch.device or str.
        cache: If True, cache a remote file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to True.
        **kwargs: Keyword arguments forwarded to torch.load.

    Returns:
        Object stored in path.
    """
    is_local = os.path.isdir(path) or os.path.isfile(path)
    if cache and not is_local:
        with fsspec.open(
            f"filecache::{path}",
            filecache={"cache_storage": str(get_user_data_dir("tts_cache"))},
            mode="rb",
        ) as f:
            return torch.load(f, map_location=map_location, **kwargs)
    else:
        with fsspec.open(path, "rb") as f:
            return torch.load(f, map_location=map_location, **kwargs)


def load_checkpoint(
    model, checkpoint_path, use_cuda=False, eval=False, cache=False
):  # pylint: disable=redefined-builtin
    try:
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), pickle_module=pickle_tts, cache=cache)
    model.load_state_dict(state["model"])
    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state
