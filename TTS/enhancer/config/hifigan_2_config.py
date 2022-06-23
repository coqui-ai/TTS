from dataclasses import dataclass, field

from TTS.enhancer.config.shared_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.hifigan_2 import HifiGAN2Args


@dataclass
class Hifigan2Config(BaseEnhancerConfig):
    """Defines parameters for the HifiGAN 2 denoiser."""

    model_args: HifiGAN2Args = field(default_factory=HifiGAN2Args)
    detach_predictor_output: bool = True
    condition_on_GT_MFCC: bool = False
    target_sr: int = 16000
    input_sr: int = 16000
    segment_len: float = 2.5
    gt_augment: bool = True
