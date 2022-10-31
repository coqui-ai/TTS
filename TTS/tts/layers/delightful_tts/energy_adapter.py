from typing import Callable, Tuple
import torch
import torch.nn as nn

from TTS.tts.layers.delightful_tts.variance_predictor import VariancePredictor
from TTS.tts.utils.helpers import average_over_durations


class EnergyAdaptor(nn.Module):
    def __init__(self, config: "AcousticModelConfig"):
        super().__init__()
        self.energy_predictor = VariancePredictor(
            channels_in=config.encoder.n_hidden,
            channels=config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=config.variance_adaptor.kernel_size,
            p_dropout=config.variance_adaptor.p_dropout,
            lrelu_slope=config.variance_adaptor.lrelu_slope
        )
        self.energy_emb = nn.Conv1d(
            1,
            config.encoder.n_hidden,
            kernel_size=config.variance_adaptor.emb_kernel_size,
            padding=int((config.variance_adaptor.emb_kernel_size - 1) / 2),
        )

    def get_energy_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, dr: torch.IntTensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
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
