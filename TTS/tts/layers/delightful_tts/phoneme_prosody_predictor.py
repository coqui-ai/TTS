from typing import Any, Dict
import torch
import torch.nn as nn

from TTS.tts.layers.delightful_tts.encoders import ReferenceEncoder
from TTS.tts.layers.delightful_tts.conformer import ConformerMultiHeadedSelfAttention
from TTS.tts.layers.delightful_tts.networks import ConvTransposed

class PhonemeProsodyPredictor(nn.Module):
    """Non-parallel Prosody Predictor inspired by Du et al., 2021"""

    def __init__(self, args: Dict[str, Any], phoneme_level: bool):
        super().__init__()
        self.d_model = args.encoder.n_hidden
        kernel_size = args.reference_encoder.predictor_kernel_size
        dropout = args.encoder.p_dropout
        bottleneck_size = (
            args.reference_encoder.bottleneck_size_p if phoneme_level else args.reference_encoder.bottleneck_size_u
        )
        self.layers = nn.ModuleList(
            [
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(args.variance_adaptor.lrelu_slope),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
                ConvTransposed(
                    self.d_model,
                    self.d_model,
                    kernel_size=kernel_size,
                    padding=(kernel_size - 1) // 2,
                ),
                nn.LeakyReLU(args.variance_adaptor.lrelu_slope),
                nn.LayerNorm(self.d_model),
                nn.Dropout(dropout),
            ]
        )
        self.predictor_bottleneck = nn.Linear(self.d_model, bottleneck_size)