import importlib
from typing import Dict

import torch

from TTS.utils.training import NoamLR


def is_apex_available():
    return importlib.util.find_spec("apex") is not None


def setup_torch_training_env(cudnn_enable, cudnn_benchmark):
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
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
    optimizer_name: str, optimizer_params: dict, lr: float, model: torch.nn.Module
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
    return optimizer(model.parameters(), lr=lr, **optimizer_params)
