from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING

from TTS.enhancer.config.shared_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.bwe import BWEArgs


@dataclass
class BWEConfig(BaseEnhancerConfig):
    """Defines parameters for a Generic Enhancer model."""

    model_args: BWEArgs = field(default_factory=BWEArgs)
    target_sr: int = 48000
    input_sr: int = 16000
    segment_train: bool = True
    segment_len: float = 1.0