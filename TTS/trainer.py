# -*- coding: utf-8 -*-

import importlib
from abc import ABC, abstractmethod
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, TypeVar, Union

import torch
from coqpit import Coqpit

# DISTRIBUTED
from torch import nn

from TTS.utils.logging import ConsoleLogger, TensorboardLogger

_DataLoader = TypeVar("_DataLoader")


@dataclass
class TrainingArgs(Coqpit):
    """Trainer arguments that are parsed externally (e.g. CLI)"""

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting best loss. If not specified, the latest best model in continue path is used"
        },
    )
    config_path: str = field(default="", metadata={"help": "Path to the configuration file."})
    rank: int = field(default=0, metadata={"help": "Process rank in distributed training."})
    group_id: str = field(default="", metadata={"help": "Process group id in distributed training."})


# pylint: disable=import-outside-toplevel, too-many-public-methods


class TrainerAbstract(ABC):
    @abstractmethod
    def __init__(
        self,
        args: Union[Coqpit, Namespace],
        config: Coqpit,
        c_logger: ConsoleLogger = None,
        tb_logger: TensorboardLogger = None,
        model: nn.Module = None,
        output_path: str = None,
    ) -> None:
        pass

    @staticmethod
    def _is_apex_available():
        return importlib.util.find_spec("apex") is not None

    @abstractmethod
    def get_model(*args, **kwargs) -> nn.Module:
        pass

    @abstractmethod
    def get_optimizer(model: nn.Module, config: Coqpit) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def get_scheduler(
        config: Coqpit, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:  # pylint: disable=protected-access
        pass

    @abstractmethod
    def get_criterion(config: Coqpit) -> nn.Module:
        pass

    @abstractmethod
    def restore_model(self, *args, **kwargs) -> Tuple:
        pass

    @abstractmethod
    def get_train_dataloader(self, *args, **kwargs) -> _DataLoader:
        pass

    @abstractmethod
    def get_eval_dataloder(self, *args, **kwargs) -> _DataLoader:
        pass

    @abstractmethod
    def format_batch(self, batch: List) -> Dict:
        pass

    @abstractmethod
    def _train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        pass

    @abstractmethod
    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        pass

    @abstractmethod
    def train_epoch(self) -> None:
        pass

    @abstractmethod
    def _eval_step(self, batch: Dict) -> Tuple[Dict, Dict]:
        pass

    @abstractmethod
    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        pass

    @abstractmethod
    def eval_epoch(self) -> None:
        pass

    @abstractmethod
    def test_run(self) -> None:
        pass

    @abstractmethod
    def fit(self) -> None:
        pass

    @abstractmethod
    def save_best_model(self) -> None:
        pass

    @abstractmethod
    def on_epoch_start(self) -> None:
        pass

    @abstractmethod
    def on_epoch_end(self) -> None:
        pass

    @abstractmethod
    def on_train_step_start(self) -> None:
        pass

    @abstractmethod
    def on_train_step_end(self) -> None:
        pass
