from dataclasses import dataclass
from typing import List

from .shared_configs import BaseTTSConfig, GSTConfig


@dataclass
class TacotronConfig(BaseTTSConfig):
    """Defines parameters for Tacotron based models."""

    model: str = "tacotron"
    use_gst: bool = False
    gst: GSTConfig = None
    gst_style_input: str = None
    # model specific params
    r: int = 2
    gradual_training: List = None
    memory_size: int = -1
    prenet_type: str = "original"
    prenet_dropout: bool = True
    prenet_dropout_at_inference: bool = False
    stopnet: bool = True
    separate_stopnet: bool = True
    stopnet_pos_weight: float = 10.0

    # attention layers
    attention_type: str = "original"
    attention_heads: int = None
    attention_norm: str = "sigmoid"
    windowing: bool = False
    use_forward_attn: bool = False
    forward_attn_mask: bool = False
    transition_agent: bool = False
    location_attn: bool = True

    # advance methods
    bidirectional_decoder: bool = False
    double_decoder_consistency: bool = False
    ddc_r: int = 6

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
    seq_len_norm: bool = False
    loss_masking: bool = True

    # loss params
    decoder_loss_alpha: float = 0.25
    postnet_loss_alpha: float = 0.25
    postnet_diff_spec_alpha: float = 0.25
    decoder_diff_spec_alpha: float = 0.25
    decoder_ssim_alpha: float = 0.25
    postnet_ssim_alpha: float = 0.25
    ga_alpha: float = 5.0
