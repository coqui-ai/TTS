from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.forward_tts_e2e import ForwardTTSE2eArgs


@dataclass
class FastPitchE2eConfig(BaseTTSConfig):
    """Configure `ForwardTTSE2e` as FastPitchE2e model.

    Example:

        >>> from TTS.tts.configs.fast_pitch_e2e_config import FastPitchE2EConfig
        >>> config = FastPitchE2EConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `fast_pitch`.

        base_model (str):
            Name of the base model being configured as this model so that 🐸 TTS knows it needs to initiate
            the base model rather than searching for the `model` implementation. Defaults to `forward_tts`.

        model_args (Coqpit):
            Model class arguments. Check `FastPitchArgs` for more details. Defaults to `FastPitchArgs()`.

        data_dep_init_steps (int):
            Number of steps used for computing normalization parameters at the beginning of the training. GlowTTS uses
            Activation Normalization that pre-computes normalization stats at the beginning and use the same values
            for the rest. Defaults to 10.

        speakers_file (str):
            Path to the file containing the list of speakers. Needed at inference for loading matching speaker ids to
            speaker names. Defaults to `None`.

        use_speaker_embedding (bool):
            enable / disable using speaker embeddings for multi-speaker models. If set True, the model is
            in the multi-speaker mode. Defaults to False.

        use_d_vector_file (bool):
            enable /disable using external speaker embeddings in place of the learned embeddings. Defaults to False.

        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Dimension of the external speaker embeddings. Defaults to 0.

        optimizer (str):
            Name of the model optimizer. Defaults to `Adam`.

        optimizer_params (dict):
            Arguments of the model optimizer. Defaults to `{"betas": [0.9, 0.998], "weight_decay": 1e-6}`.

        lr_scheduler (str):
            Name of the learning rate scheduler. Defaults to `Noam`.

        lr_scheduler_params (dict):
            Arguments of the learning rate scheduler. Defaults to `{"warmup_steps": 4000}`.

        lr (float):
            Initial learning rate. Defaults to `1e-3`.

        grad_clip (float):
            Gradient norm clipping value. Defaults to `5.0`.

        spec_loss_type (str):
            Type of the spectrogram loss. Check `ForwardTTSLoss` for possible values. Defaults to `mse`.

        duration_loss_type (str):
            Type of the duration loss. Check `ForwardTTSLoss` for possible values. Defaults to `mse`.

        use_ssim_loss (bool):
            Enable/disable the use of SSIM (Structural Similarity) loss. Defaults to True.

        wd (float):
            Weight decay coefficient. Defaults to `1e-7`.

        ssim_loss_alpha (float):
            Weight for the SSIM loss. If set 0, disables the SSIM loss. Defaults to 1.0.

        dur_loss_alpha (float):
            Weight for the duration predictor's loss. If set 0, disables the huber loss. Defaults to 1.0.

        spec_loss_alpha (float):
            Weight for the L1 spectrogram loss. If set 0, disables the L1 loss. Defaults to 1.0.

        pitch_loss_alpha (float):
            Weight for the pitch predictor's loss. If set 0, disables the pitch predictor. Defaults to 1.0.

        binary_align_loss_alpha (float):
            Weight for the binary loss. If set 0, disables the binary loss. Defaults to 1.0.

        binary_loss_warmup_epochs (float):
            Number of epochs to gradually increase the binary loss impact. Defaults to 150.

        min_seq_len (int):
            Minimum input sequence length to be used at training.

        max_seq_len (int):
            Maximum input sequence length to be used at training. Larger values result in more VRAM usage.
    """

    model: str = "fast_pitch_e2e"
    base_model: str = "forward_tts_e2e"

    # model specific params
    # model_args: ForwardTTSE2eArgs = ForwardTTSE2eArgs(vocoder_config=HifiganConfig())
    model_args: ForwardTTSE2eArgs = ForwardTTSE2eArgs()

    # multi-speaker settings
    # num_speakers: int = 0
    # speakers_file: str = None
    # use_speaker_embedding: bool = False
    # use_d_vector_file: bool = False
    # d_vector_file: str = False
    # d_vector_dim: int = 0
    spec_segment_size: int = 30

    # optimizer
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

    # encoder loss params
    spec_loss_type: str = "mse"
    duration_loss_type: str = "mse"
    use_ssim_loss: bool = False
    ssim_loss_alpha: float = 0.0
    spec_loss_alpha: float = 1.0
    aligner_loss_alpha: float = 1.0
    pitch_loss_alpha: float = 1.0
    dur_loss_alpha: float = 1.0
    binary_align_loss_alpha: float = 0.1
    binary_loss_warmup_epochs: int = 150

    # dvocoder loss params
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 10.0
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

    # overrides
    r: int = 1

    # dataset configs
    compute_f0: bool = True
    f0_cache_path: str = None

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
