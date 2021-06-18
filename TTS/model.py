from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from coqpit import Coqpit
from torch import nn

from TTS.utils.audio import AudioProcessor

# pylint: skip-file


class BaseModel(nn.Module, ABC):
    """Abstract 🐸TTS class. Every new 🐸TTS model must inherit this.

    Notes on input/output tensor shapes:
        Any input or output tensor of the model must be shaped as

        - 3D tensors `batch x time x channels`
        - 2D tensors `batch x channels`
        - 1D tensors `batch x 1`
    """

    @abstractmethod
    def forward(self, text: torch.Tensor, aux_input={}, **kwargs) -> Dict:
        """Forward pass for the model mainly used in training.

        You can be flexible here and use different number of arguments and argument names since it is mostly used by
        `train_step()` in training whitout exposing it to the out of the class.

        Args:
            text (torch.Tensor): Input text character sequence ids.
            aux_input (Dict): Auxiliary model inputs like embeddings, durations or any other sorts of inputs.
                for the model.

        Returns:
            Dict: model outputs. This must include an item keyed `model_outputs` as the final artifact of the model.
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    @abstractmethod
    def inference(self, text: torch.Tensor, aux_input={}) -> Dict:
        """Forward pass for inference.

        After the model is trained this is the only function that connects the model the out world.

        This function must only take a `text` input and a dictionary that has all the other model specific inputs.
        We don't use `*kwargs` since it is problematic with the TorchScript API.

        Args:
            text (torch.Tensor): [description]
            aux_input (Dict): Auxiliary inputs like speaker embeddings, durations etc.

        Returns:
            Dict: [description]
        """
        outputs_dict = {"model_outputs": None}
        ...
        return outputs_dict

    @abstractmethod
    def train_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        outputs_dict = {}
        loss_dict = {}  # this returns from the criterion
        ...
        return outputs_dict, loss_dict

    def train_log(self, ap: AudioProcessor, batch: Dict, outputs: Dict) -> Tuple[Dict, np.ndarray]:
        """Create visualizations and waveform examples for training.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        return None, None

    @abstractmethod
    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single evaluation step. Run the model forward pass and compute losses. In most cases, you can
        call `train_step()` with no changes.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """
        outputs_dict = {}
        loss_dict = {}  # this returns from the criterion
        ...
        return outputs_dict, loss_dict

    def eval_log(self, ap: AudioProcessor, batch: Dict, outputs: Dict) -> Tuple[Dict, np.ndarray]:
        """The same as `train_log()`"""
        return None, None

    @abstractmethod
    def load_checkpoint(self, config: Coqpit, checkpoint_path: str, eval: bool = False) -> None:
        """Load a checkpoint and get ready for training or inference.

        Args:
            config (Coqpit): Model configuration.
            checkpoint_path (str): Path to the model checkpoint file.
            eval (bool, optional): If true, init model for inference else for training. Defaults to False.
        """
        ...

    def get_optimizer(self) -> Union["Optimizer", List["Optimizer"]]:
        """Setup an return optimizer or optimizers."""
        pass

    def get_lr(self) -> Union[float, List[float]]:
        """Return learning rate(s).

        Returns:
            Union[float, List[float]]: Model's initial learning rates.
        """
        pass

    def get_scheduler(self, optimizer: torch.optim.Optimizer):
        pass

    def get_criterion(self):
        pass

    def format_batch(self):
        pass
