from dataclasses import asdict, dataclass, field
from typing import Dict, List

from coqpit import MISSING

from TTS.config.shared_configs import BaseAudioConfig, BaseTrainingConfig


@dataclass
class BaseEnhancerConfig(BaseTrainingConfig):
    """Defines parameters for a Generic Encoder model."""

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    datasets: List[str] = None
    eval_split_max_size: int = None
    eval_split_size: float = 0.01
    target_sr: int = 48000
    input_sr: int = 16000
    segment_train: bool = True
    gt_augment: bool = False
    segment_len: float = 1.0
    grad_clip: List[float] = field(default_factory=lambda: [3.0, 3.0])
    audio_augmentation: Dict = field(default_factory=lambda: {})
    # optimizer
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})
    target_loss: str = "loss_1"
    # scheduler
    lr_disc: float = 0.001
    lr_gen: float = 0.001
    lr_scheduler_gen: str = None
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {})
    lr_scheduler_disc: str = None
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {})
    # gan
    steps_to_start_discriminator: int = 50_000
