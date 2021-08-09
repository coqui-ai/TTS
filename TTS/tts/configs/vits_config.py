from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.vits import VitsArgs


@dataclass
class VitsConfig(BaseTTSConfig):
    """Defines parameters for VITS End2End TTS model.

    Example:

        >>> from TTS.tts.configs import VitsConfig
        >>> config = VitsConfig()
    """

    model: str = "vits"
    # model specific params
    model_args: VitsArgs = field(default_factory=VitsArgs)

    # optimizer
    grad_clip: float = field(default_factory=lambda: [5, 5])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # loss params
    kl_loss_alpha: float = 1.0
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 45.0

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # testing
    test_sentences: List[str] = field(
        default_factory=lambda: [
            "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
            "Be a voice, not an echo.",
            "I'm sorry Dave. I'm afraid I can't do that.",
            "This cake is great. It's so delicious and moist.",
            "Prior to November 22, 1963.",
        ]
    )
