from dataclasses import asdict, dataclass, field
from typing import Dict, List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, DelightfulTtsAudioConfig, VocoderConfig


@dataclass
class DelightfulTTSConfig(BaseTTSConfig):

    model: str = "delightful_tts"

    # model specific params
    audio: DelightfulTtsAudioConfig = DelightfulTtsAudioConfig()
    model_args: DelightfulTtsArgs = DelightfulTtsArgs()
    use_attn_priors: bool = True

    # vocoder
    vocoder: VocoderConfig = VocoderConfig()
    init_discriminator: bool = True

    # optimizer
    steps_to_start_discriminator: int = 200000
    grad_clip: List[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # acoustic model loss params
    ssim_loss_alpha: float = 1.0
    mel_loss_alpha: float = 1.0
    aligner_loss_alpha: float = 1.0
    pitch_loss_alpha: float = 1.0
    energy_loss_alpha: float = 1.0
    u_prosody_loss_alpha: float = 0.5
    p_prosody_loss_alpha: float = 0.5
    dur_loss_alpha: float = 1.0
    char_dur_loss_alpha: float = 0.01
    binary_align_loss_alpha: float = 0.1
    binary_loss_warmup_epochs: int = 10

    # vocoder loss params
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    vocoder_mel_loss_alpha: float = 10.0
    multi_scale_stft_loss_alpha: float = 2.5
    multi_scale_stft_loss_params: dict = field(
        default_factory=lambda: {
            "n_ffts": [1024, 2048, 512],
            "hop_lengths": [120, 240, 50],
            "win_lengths": [600, 1200, 240],
        }
    )

    # data loader params
    return_wav: bool = True
    use_weighted_sampler: bool = False
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1

    # dataset configs
    compute_f0: bool = True
    f0_cache_path: str = None
    attn_prior_cache_path: str = None

    # multi-speaker settings
    # use speaker embedding layer
    num_speakers: int = 0
    use_speaker_embedding: bool = False
    speakers_file: str = None
    speaker_embedding_channels: int = 256
    language_ids_file: str = None
    use_language_embedding: bool = False

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: str = None
    d_vector_dim: int = None

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
    
    def __post_init__(self):
        # Pass multi-speaker parameters to the model args as `model.init_multispeaker()` looks for it there.
        if self.num_speakers > 0:
            self.model_args.num_speakers = self.num_speakers

        # speaker embedding settings
        if self.use_speaker_embedding:
            self.model_args.use_speaker_embedding = True
        if self.speakers_file:
            self.model_args.speakers_file = self.speakers_file

        # d-vector settings
        if self.use_d_vector_file:
            self.model_args.use_d_vector_file = True
        if self.d_vector_dim is not None and self.d_vector_dim > 0:
            self.model_args.d_vector_dim = self.d_vector_dim
        if self.d_vector_file:
            self.model_args.d_vector_file = self.d_vector_file
