from dataclasses import dataclass, field

from TTS.enhancer.configs.shared_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.hifigan_2 import Hifigan2Args


@dataclass
class Hifigan2Config(BaseEnhancerConfig):
    """Defines parameters for the Hifigan 2 denoiser."""

    model: str = "hifigan_denoiser_2"
    model_args: Hifigan2Args = field(default_factory=Hifigan2Args)
    detach_predictor_output: bool = True
    condition_on_GT_MFCC: bool = False
    target_sr: int = 16000
    input_sr: int = 16000
    segment_len: float = 2.5
    gt_augment: bool = True
