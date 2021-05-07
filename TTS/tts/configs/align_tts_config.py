from dataclasses import dataclass, field

from .shared_configs import BaseTTSConfig


@dataclass
class AlignTTSConfig(BaseTTSConfig):
    """Defines parameters for AlignTTS model."""

    model: str = "align_tts"
    # model specific params
    positional_encoding: bool = True
    hidden_channels_dp: int = 256
    hidden_channels: int = 256
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    phase_start_steps: list = None

    ssim_alpha: float = 1.0
    spec_loss_alpha: float = 1.0
    dur_loss_alpha: float = 1.0
    mdn_alpha: float = 1.0

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

    # overrides
    min_seq_len: int = 13
    max_seq_len: int = 200
    r: int = 1
