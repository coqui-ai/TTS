from dataclasses import dataclass, field

from .shared_configs import BaseTTSConfig


@dataclass
class SpeedySpeechConfig(BaseTTSConfig):
    """Defines parameters for Speedy Speech (feed-forward encoder-decoder) based models."""

    model: str = "speedy_speech"
    # model specific params
    positional_encoding: bool = True
    hidden_channels: int = 128
    encoder_type: str = "residual_conv_bn"
    encoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 4,
            "dilations": [1, 2, 4, 1, 2, 4, 1, 2, 4, 1, 2, 4, 1],
            "num_conv_blocks": 2,
            "num_res_blocks": 13,
        }
    )
    decoder_type: str = "residual_conv_bn"
    decoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 4,
            "dilations": [1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1, 2, 4, 8, 1],
            "num_conv_blocks": 2,
            "num_res_blocks": 17,
        }
    )

    # multi-speaker settings
    use_speaker_embedding: bool = False
    use_external_speaker_embedding_file: bool = False
    external_speaker_embedding_file: str = False

    # optimizer parameters
    noam_schedule: bool = False
    warmup_steps: int = 4000
    lr: float = 1e-4
    wd: float = 1e-6
    grad_clip: float = 5.0

    # loss params
    ssim_alpha: float = 1.0
    huber_alpha: float = 1.0
    l1_alpha: float = 1.0

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1
