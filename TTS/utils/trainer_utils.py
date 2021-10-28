import importlib
import os
import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

import fsspec
import torch

from TTS.utils.io import load_fsspec
from TTS.utils.training import NoamLR


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def setup_torch_training_env(cudnn_enable: bool, cudnn_benchmark: bool, use_ddp: bool = False) -> Tuple[bool, int]:
    """Setup PyTorch environment for training.

    Args:
        cudnn_enable (bool): Enable/disable CUDNN.
        cudnn_benchmark (bool): Enable/disable CUDNN benchmarking. Better to set to False if input sequence length is
            variable between batches.
        use_ddp (bool): DDP flag. True if DDP is enabled, False otherwise.

    Returns:
        Tuple[bool, int]: is cuda on or off and number of GPUs in the environment.
    """
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1 and not use_ddp:
        raise RuntimeError(
            f" [!] {num_gpus} active GPUs. Define the target GPU by `CUDA_VISIBLE_DEVICES`. For multi-gpu training use `TTS/bin/distribute.py`."
        )
    torch.backends.cudnn.enabled = cudnn_enable
    torch.backends.cudnn.benchmark = cudnn_benchmark
    torch.manual_seed(54321)
    use_cuda = torch.cuda.is_available()
    print(" > Using CUDA: ", use_cuda)
    print(" > Number of GPUs: ", num_gpus)
    return use_cuda, num_gpus


def get_scheduler(
    lr_scheduler: str, lr_scheduler_params: Dict, optimizer: torch.optim.Optimizer
) -> torch.optim.lr_scheduler._LRScheduler:  # pylint: disable=protected-access
    """Find, initialize and return a scheduler.

    Args:
        lr_scheduler (str): Scheduler name.
        lr_scheduler_params (Dict): Scheduler parameters.
        optimizer (torch.optim.Optimizer): Optimizer to pass to the scheduler.

    Returns:
        torch.optim.lr_scheduler._LRScheduler: Functional scheduler.
    """
    if lr_scheduler is None:
        return None
    if lr_scheduler.lower() == "noamlr":
        scheduler = NoamLR
    else:
        scheduler = getattr(torch.optim.lr_scheduler, lr_scheduler)
    return scheduler(optimizer, **lr_scheduler_params)


def get_optimizer(
    optimizer_name: str, optimizer_params: dict, lr: float, model: torch.nn.Module = None, parameters: List = None
) -> torch.optim.Optimizer:
    """Find, initialize and return a optimizer.

    Args:
        optimizer_name (str): Optimizer name.
        optimizer_params (dict): Optimizer parameters.
        lr (float): Initial learning rate.
        model (torch.nn.Module): Model to pass to the optimizer.

    Returns:
        torch.optim.Optimizer: Functional optimizer.
    """
    if optimizer_name.lower() == "radam":
        module = importlib.import_module("TTS.utils.radam")
        optimizer = getattr(module, "RAdam")
    else:
        optimizer = getattr(torch.optim, optimizer_name)
    if model is not None:
        parameters = model.parameters()
    return optimizer(parameters, lr=lr, **optimizer_params)


def get_last_checkpoint(path: str) -> Tuple[str, str]:
    """Get latest checkpoint or/and best model in path.

    It is based on globbing for `*.pth.tar` and the RegEx
    `(checkpoint|best_model)_([0-9]+)`.

    Args:
        path: Path to files to be compared.

    Raises:
        ValueError: If no checkpoint or best_model files are found.

    Returns:
        Path to the last checkpoint
        Path to best checkpoint
    """
    fs = fsspec.get_mapper(path).fs
    file_names = fs.glob(os.path.join(path, "*.pth.tar"))
    scheme = urlparse(path).scheme
    if scheme:  # scheme is not preserved in fs.glob, add it back
        file_names = [scheme + "://" + file_name for file_name in file_names]
    last_models = {}
    last_model_nums = {}
    for key in ["checkpoint", "best_model"]:
        last_model_num = None
        last_model = None
        # pass all the checkpoint files and find
        # the one with the largest model number suffix.
        for file_name in file_names:
            match = re.search(f"{key}_([0-9]+)", file_name)
            if match is not None:
                model_num = int(match.groups()[0])
                if last_model_num is None or model_num > last_model_num:
                    last_model_num = model_num
                    last_model = file_name

        # if there is no checkpoint found above
        # find the checkpoint with the latest
        # modification date.
        key_file_names = [fn for fn in file_names if key in fn]
        if last_model is None and len(key_file_names) > 0:
            last_model = max(key_file_names, key=os.path.getctime)
            last_model_num = load_fsspec(last_model)["step"]

        if last_model is not None:
            last_models[key] = last_model
            last_model_nums[key] = last_model_num

    # check what models were found
    if not last_models:
        raise ValueError(f"No models found in continue path {path}!")
    if "checkpoint" not in last_models:  # no checkpoint just best model
        last_models["checkpoint"] = last_models["best_model"]
    elif "best_model" not in last_models:  # no best model
        # this shouldn't happen, but let's handle it just in case
        last_models["best_model"] = last_models["checkpoint"]
    # finally check if last best model is more recent than checkpoint
    elif last_model_nums["best_model"] > last_model_nums["checkpoint"]:
        last_models["checkpoint"] = last_models["best_model"]

    return last_models["checkpoint"], last_models["best_model"]
