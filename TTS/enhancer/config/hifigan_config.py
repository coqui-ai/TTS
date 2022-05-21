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
    steps_to_start_discriminator: int = 100_000
    steps_to_start_postnet: int = 20_000
    gt_augment: bool = True
    lr_gen: float = 0.002
    lr_disc: float = 0.02
    lr_scheduler_gen: str = "StepLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"step_size": 30, "gamma": 0.1})
    lr_scheduler_disc: str = "StepLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"step_size": 30, "gamma": 0.1})

