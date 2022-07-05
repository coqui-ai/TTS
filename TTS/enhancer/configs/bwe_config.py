from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING

from TTS.enhancer.configs.shared_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.bwe import BweArgs


@dataclass
class BweConfig(BaseEnhancerConfig):
    """Defines parameters for a Generic Enhancer model."""

    model: str = "bwe"
    model_args: BweArgs = field(default_factory=BweArgs)
    target_sr: int = 48000
    input_sr: int = 16000
    segment_train: bool = True
    segment_len: float = 1.0
