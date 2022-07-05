from dataclasses import dataclass, field

from TTS.enhancer.configs.shared_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.hifigan import HifiganArgs


@dataclass
class HifiganConfig(BaseEnhancerConfig):
    """Defines parameters for the Hifigan denoiser."""

    model: str = "hifigan_denoiser"
    model_args: HifiganArgs = field(default_factory=HifiganArgs)
    target_sr: int = 16000
    input_sr: int = 16000
    segment_len: float = 2
    steps_to_start_discriminator: int = 40_000
    steps_to_start_postnet: int = 20_000
    gt_augment: bool = True
