import datetime
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


def save_fsspec(state: Any, path: str, **kwargs):
    """Like torch.save but can save to other locations (e.g. s3:// , gs://).

    Args:
        state: State object to save
        path: Any path or url supported by fsspec.
        **kwargs: Keyword arguments forwarded to torch.save.
    """
    with fsspec.open(path, "wb") as f:
        torch.save(state, f, **kwargs)


def save_model(config, model, optimizer, scaler, current_step, epoch, output_path, **kwargs):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    if isinstance(optimizer, list):
        optimizer_state = [optim.state_dict() for optim in optimizer]
    elif optimizer.__class__.__name__ == "CapacitronOptimizer":
        optimizer_state = [optimizer.primary_optimizer.state_dict(), optimizer.secondary_optimizer.state_dict()]
    else:
        optimizer_state = optimizer.state_dict() if optimizer is not None else None

    if isinstance(scaler, list):
        scaler_state = [s.state_dict() for s in scaler]
    else:
        scaler_state = scaler.state_dict() if scaler is not None else None

    if isinstance(config, Coqpit):
        config = config.to_dict()

    state = {
        "config": config,
        "model": model_state,
        "optimizer": optimizer_state,
        "scaler": scaler_state,
        "step": current_step,
        "epoch": epoch,
        "date": datetime.date.today().strftime("%B %d, %Y"),
    }
    state.update(kwargs)
    save_fsspec(state, output_path)


def save_checkpoint(
    config,
    model,
    optimizer,
    scaler,
    current_step,
    epoch,
    output_folder,
    **kwargs,
):
    file_name = "checkpoint_{}.pth".format(current_step)
    checkpoint_path = os.path.join(output_folder, file_name)
    print("\n > CHECKPOINT : {}".format(checkpoint_path))
    save_model(
        config,
        model,
        optimizer,
        scaler,
        current_step,
        epoch,
        checkpoint_path,
        **kwargs,
    )


def save_best_model(
    current_loss,
    best_loss,
    config,
    model,
    optimizer,
    scaler,
    current_step,
    epoch,
    out_path,
    keep_all_best=False,
    keep_after=10000,
    **kwargs,
):
    if current_loss < best_loss:
        best_model_name = f"best_model_{current_step}.pth"
        checkpoint_path = os.path.join(out_path, best_model_name)
        print(" > BEST MODEL : {}".format(checkpoint_path))
        save_model(
            config,
            model,
            optimizer,
            scaler,
            current_step,
            epoch,
            checkpoint_path,
            model_loss=current_loss,
            **kwargs,
        )
        fs = fsspec.get_mapper(out_path).fs
        # only delete previous if current is saved successfully
        if not keep_all_best or (current_step < keep_after):
            model_names = fs.glob(os.path.join(out_path, "best_model*.pth"))
            for model_name in model_names:
                if os.path.basename(model_name) != best_model_name:
                    fs.rm(model_name)
        # create a shortcut which always points to the currently best model
        shortcut_name = "best_model.pth"
        shortcut_path = os.path.join(out_path, shortcut_name)
        fs.copy(checkpoint_path, shortcut_path)
        best_loss = current_loss
    return best_loss
