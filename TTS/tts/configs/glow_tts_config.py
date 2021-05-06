from dataclasses import dataclass, field

from .shared_configs import BaseTTSConfig


@dataclass
class GlowTTSConfig(BaseTTSConfig):
    """Defines parameters for GlowTTS model."""

    model: str = "glow_tts"

    # model params
    encoder_type: str = "rel_pos_transformer"
    encoder_params: dict = field(
        default_factory=lambda: {
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,
            "num_heads": 2,
            "hidden_channels_ffn": 768,
        }
    )
    use_encoder_prenet: bool = True
    hidden_channels_encoder: int = 192
    hidden_channels_decoder: int = 192
    hidden_channels_duration_predictor: int = 256

    # training params
    data_dep_init_steps: int = 10

    # inference params
    style_wav_for_test: str = None
    inference_noise_scale: float = 0.0

    # multi-speaker settings
    use_speaker_embedding: bool = False
    use_external_speaker_embedding_file: bool = False
    external_speaker_embedding_file: str = False

    # optimizer params
    noam_schedule: bool = True
    warmup_steps: int = 4000
    grad_clip: float = 5.0
    lr: float = 1e-3
    wd: float = 0.000001

    # overrides
    min_seq_len: int = 3
    max_seq_len: int = 500
    r: int = 1
