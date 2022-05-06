from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig
from TTS.enhancer.models.bwe import BWEArgs


@dataclass
class BaseEnhancerConfig(BaseTrainingConfig):
    """Defines parameters for a Generic Encoder model."""

    model_args: BWEArgs = field(default_factory=BWEArgs)
    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    target_sr: int = 48000
    input_sr: int = 16000
    segment_train: bool = True
    segment_len: float = 1.0
    grad_clip: List[float] = field(default_factory=lambda: [3.0, 3.0])
    # model params
    audio_augmentation: Dict = field(default_factory=lambda: {})
    # optimizer
    optimizer: str = "Adam"
    optimizer_params: dict = field(default_factory=lambda: {})
    target_loss: str = "loss_0"
    # scheduler
    lr_disc: float = 0.001
    lr_gen: float = 0.0001
    lr_scheduler: str = None
    lr_scheduler_params: dict = field(default_factory=lambda: {})
    # gan
    steps_to_start_discriminator: int = 50000
