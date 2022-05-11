from dataclasses import dataclass, field

from TTS.enhancer.config.shared_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.hifigan import HifiGANArgs


@dataclass
class HifiganConfig(BaseEnhancerConfig):
    """Defines parameters for the HifiGAN denoiser."""

    model_args: HifiGANArgs = field(default_factory=HifiGANArgs)
    target_sr: int = 16000
    input_sr: int = 16000
    segment_len: float = 1.5

