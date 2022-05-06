from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING
from torch import detach

from TTS.enhancer.config.base_enhancer_config import BaseEnhancerConfig


@dataclass
class Hifigan2Config(BaseEnhancerConfig):
    """Defines parameters for the HifiGAN 2 denoiser."""

    detach_predictor_output: bool = True
