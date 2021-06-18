import datetime
import glob
import os
import pickle as pickle_tts
from shutil import copyfile

import torch
from coqpit import Coqpit


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


def load_checkpoint(model, checkpoint_path, use_cuda=False, eval=False):  # pylint: disable=redefined-builtin
    try:
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    except ModuleNotFoundError:
        pickle_tts.Unpickler = RenamingUnpickler
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"), pickle_module=pickle_tts)
    model.load_state_dict(state["model"])
    if use_cuda:
        model.cuda()
    if eval:
        model.eval()
    return model, state


def save_model(config, model, optimizer, scaler, current_step, epoch, output_path, **kwargs):
    if hasattr(model, "module"):
        model_state = model.module.state_dict()
    else:
        model_state = model.state_dict()
    if isinstance(optimizer, list):
        optimizer_state = [optim.state_dict() for optim in optimizer]
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
    torch.save(state, output_path)


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
    file_name = "checkpoint_{}.pth.tar".format(current_step)
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
        best_model_name = f"best_model_{current_step}.pth.tar"
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
        # only delete previous if current is saved successfully
        if not keep_all_best or (current_step < keep_after):
            model_names = glob.glob(os.path.join(out_path, "best_model*.pth.tar"))
            for model_name in model_names:
                if os.path.basename(model_name) == best_model_name:
                    continue
                os.remove(model_name)
        # create symlink to best model for convinience
        link_name = "best_model.pth.tar"
        link_path = os.path.join(out_path, link_name)
        if os.path.islink(link_path) or os.path.isfile(link_path):
            os.remove(link_path)
        os.symlink(best_model_name, os.path.join(out_path, link_name))
        best_loss = current_loss
    return best_loss
