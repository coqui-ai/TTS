from typing import Callable, Tuple

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import

from TTS.tts.layers.delightful_tts.variance_predictor import VariancePredictor
from TTS.tts.utils.helpers import average_over_durations


class EnergyAdaptor(nn.Module):  # pylint: disable=abstract-method
    """Variance Adaptor with an added 1D conv layer. Used to
    get energy embeddings.

    Args:
        channels_in (int): Number of in channels for conv layers.
        channels_out (int): Number of out channels.
        kernel_size (int): Size the kernel for the conv layers.
        dropout (float): Probability of dropout.
        lrelu_slope (float): Slope for the leaky relu.
        emb_kernel_size (int): Size the kernel for the pitch embedding.

    Inputs: inputs, mask
        - **inputs** (batch, time1, dim): Tensor containing input vector
        - **target** (batch, 1, time2): Tensor containing the energy target
        - **dr** (batch, time1): Tensor containing aligner durations vector
        - **mask** (batch, time1): Tensor containing indices to be masked
    Returns:
        - **energy prediction** (batch, 1, time1): Tensor produced by energy predictor
        - **energy embedding** (batch, channels, time1): Tensor produced energy adaptor
        - **average energy target(train only)** (batch, 1, time1): Tensor produced after averaging over durations

    """

    def __init__(
        self,
        channels_in: int,
        channels_hidden: int,
        channels_out: int,
        kernel_size: int,
        dropout: float,
        lrelu_slope: float,
        emb_kernel_size: int,
    ):
        super().__init__()
        self.energy_predictor = VariancePredictor(
            channels_in=channels_in,
            channels=channels_hidden,
            channels_out=channels_out,
            kernel_size=kernel_size,
            p_dropout=dropout,
            lrelu_slope=lrelu_slope,
        )
        self.energy_emb = nn.Conv1d(
            1,
            channels_hidden,
            kernel_size=emb_kernel_size,
            padding=int((emb_kernel_size - 1) / 2),
        )

    def get_energy_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, dr: torch.IntTensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shapes:
            x: :math: `[B, T_src, C]`
            target: :math: `[B, 1, T_max2]`
            dr: :math: `[B, T_src]`
            mask: :math: `[B, T_src]`
        """
        energy_pred = self.energy_predictor(x, mask)
        energy_pred.unsqueeze_(1)
        avg_energy_target = average_over_durations(target, dr)
        energy_emb = self.energy_emb(avg_energy_target)
        return energy_pred, avg_energy_target, energy_emb

    def get_energy_embedding(self, x: torch.Tensor, mask: torch.Tensor, energy_transform: Callable) -> torch.Tensor:
        energy_pred = self.energy_predictor(x, mask)
        energy_pred.unsqueeze_(1)
        if energy_transform is not None:
            energy_pred = energy_transform(energy_pred, (~mask).sum(dim=(1, 2)), self.pitch_mean, self.pitch_std)
        energy_emb_pred = self.energy_emb(energy_pred)
        return energy_emb_pred, energy_pred
