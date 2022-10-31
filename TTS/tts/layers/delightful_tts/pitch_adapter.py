from typing import Tuple, Callable

import torch
import torch.nn as nn

from TTS.tts.layers.delightful_tts.variance_predictor import VariancePredictor
from TTS.tts.utils.helpers import average_over_durations


class PitchAdaptor(nn.Module):
    def __init__(self, config: "AcousticModelConfig"):
        super().__init__()
        self.pitch_predictor = VariancePredictor(
            channels_in=config.encoder.n_hidden,
            channels=config.variance_adaptor.n_hidden,
            channels_out=1,
            kernel_size=config.variance_adaptor.kernel_size,
            p_dropout=config.variance_adaptor.p_dropout,
            lrelu_slope=config.variance_adaptor.lrelu_slope
        )
        self.pitch_emb = nn.Conv1d(
            1,
            config.encoder.n_hidden,
            kernel_size=config.variance_adaptor.emb_kernel_size,
            padding=int((config.variance_adaptor.emb_kernel_size - 1) / 2),
        )

    def get_pitch_embedding_train(
        self, x: torch.Tensor, target: torch.Tensor, dr: torch.IntTensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pitch_pred = self.pitch_predictor(x, mask)  # [B, T_src, C_hidden], [B, T_src] --> [B, T_src]
        pitch_pred.unsqueeze_(1)  # --> [B, 1, T_src]
        avg_pitch_target = average_over_durations(target, dr)  # [B, 1, T_mel], [B, T_src] --> [B, 1, T_src]
        pitch_emb = self.pitch_emb(avg_pitch_target)  # [B, 1, T_src] --> [B, C_hidden, T_src]
        return pitch_pred, avg_pitch_target, pitch_emb

    def get_pitch_embedding(self, x: torch.Tensor, mask: torch.Tensor, pitch_transform: Callable, pitch_mean: torch.Tensor, pitch_std: torch.Tensor) -> torch.Tensor:
        pitch_pred = self.pitch_predictor(x, mask)
        if pitch_transform is not None:
            pitch_pred = pitch_transform(pitch_pred, (~mask).sum(), pitch_mean, pitch_std)
        pitch_pred.unsqueeze_(1)
        pitch_emb_pred = self.pitch_emb(pitch_pred)
        return pitch_emb_pred, pitch_pred
