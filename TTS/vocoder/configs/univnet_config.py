from dataclasses import dataclass, field

from TTS.vocoder.configs.shared_configs import BaseGANVocoderConfig


@dataclass
class UnivnetConfig(BaseGANVocoderConfig):
    """Defines parameters for UnivNet vocoder.

    Example:

        >>> from TTS.vocoder.configs import UnivNetConfig
        >>> config = UnivNetConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `UnivNet`.
        discriminator_model (str): One of the discriminators from `TTS.vocoder.models.*_discriminator`. Defaults to
            'UnivNet_discriminator`.
        generator_model (str): One of the generators from TTS.vocoder.models.*`. Every other non-GAN vocoder model is
            considered as a generator too. Defaults to `UnivNet_generator`.
        generator_model_params (dict): Parameters of the generator model. Defaults to
            `
            {
                "use_mel": True,
                "sample_rate": 22050,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": None,
            }
            `
        batch_size (int):
            Batch size used at training. Larger values use more memory. Defaults to 32.
        seq_len (int):
            Audio segment length used at training. Larger values use more memory. Defaults to 8192.
        pad_short (int):
            Additional padding applied to the audio samples shorter than `seq_len`. Defaults to 0.
        use_noise_augment (bool):
            enable / disable random noise added to the input waveform. The noise is added after computing the
            features. Defaults to True.
        use_cache (bool):
            enable / disable in memory caching of the computed features. It can cause OOM error if the system RAM is
            not large enough. Defaults to True.
        use_stft_loss (bool):
            enable / disable use of STFT loss originally used by ParallelWaveGAN model. Defaults to True.
        use_subband_stft (bool):
            enable / disable use of subband loss computation originally used by MultiBandMelgan model. Defaults to True.
        use_mse_gan_loss (bool):
            enable / disable using Mean Squeare Error GAN loss. Defaults to True.
        use_hinge_gan_loss (bool):
            enable / disable using Hinge GAN loss. You should choose either Hinge or MSE loss for training GAN models.
            Defaults to False.
        use_feat_match_loss (bool):
            enable / disable using Feature Matching loss originally used by MelGAN model. Defaults to True.
        use_l1_spec_loss (bool):
            enable / disable using L1 spectrogram loss originally used by univnet model. Defaults to False.
        stft_loss_params (dict):
            STFT loss parameters. Default to
            `{
                "n_ffts": [1024, 2048, 512],
                "hop_lengths": [120, 240, 50],
                "win_lengths": [600, 1200, 240]
            }`
        l1_spec_loss_params (dict):
            L1 spectrogram loss parameters. Default to
            `{
                "use_mel": True,
                "sample_rate": 22050,
                "n_fft": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": None,
            }`
        stft_loss_weight (float): STFT loss weight that multiplies the computed loss before summing up the total
            model loss. Defaults to 0.5.
        subband_stft_loss_weight (float):
            Subband STFT loss weight that multiplies the computed loss before summing up the total loss. Defaults to 0.
        mse_G_loss_weight (float):
            MSE generator loss weight that multiplies the computed loss before summing up the total loss. faults to 2.5.
        hinge_G_loss_weight (float):
            Hinge generator loss weight that multiplies the computed loss before summing up the total loss. Defaults to 0.
        feat_match_loss_weight (float):
            Feature matching loss weight that multiplies the computed loss before summing up the total loss. faults to 108.
        l1_spec_loss_weight (float):
            L1 spectrogram loss weight that multiplies the computed loss before summing up the total loss. Defaults to 0.
    """

    model: str = "univnet"
    batch_size: int = 32
    # model specific params
    discriminator_model: str = "univnet_discriminator"
    generator_model: str = "univnet_generator"
    generator_model_params: dict = field(
        default_factory=lambda: {
            "in_channels": 64,
            "out_channels": 1,
            "hidden_channels": 32,
            "cond_channels": 80,
            "upsample_factors": [8, 8, 4],
            "lvc_layers_each_block": 4,
            "lvc_kernel_size": 3,
            "kpnet_hidden_channels": 64,
            "kpnet_conv_size": 3,
            "dropout": 0.0,
        }
    )

    # LOSS PARAMETERS - overrides
    use_stft_loss: bool = True
    use_subband_stft_loss: bool = False
    use_mse_gan_loss: bool = True
    use_hinge_gan_loss: bool = False
    use_feat_match_loss: bool = False  # requires MelGAN Discriminators (MelGAN and univnet)
    use_l1_spec_loss: bool = False

    # loss weights - overrides
    stft_loss_weight: float = 2.5
    stft_loss_params: dict = field(
        default_factory=lambda: {
            "n_ffts": [1024, 2048, 512],
            "hop_lengths": [120, 240, 50],
            "win_lengths": [600, 1200, 240],
        }
    )
    subband_stft_loss_weight: float = 0
    mse_G_loss_weight: float = 1
    hinge_G_loss_weight: float = 0
    feat_match_loss_weight: float = 0
    l1_spec_loss_weight: float = 0
    l1_spec_loss_params: dict = field(
        default_factory=lambda: {
            "use_mel": True,
            "sample_rate": 22050,
            "n_fft": 1024,
            "hop_length": 256,
            "win_length": 1024,
            "n_mels": 80,
            "mel_fmin": 0.0,
            "mel_fmax": None,
        }
    )

    # optimizer parameters
    lr_gen: float = 1e-4  # Initial learning rate.
    lr_disc: float = 1e-4  # Initial learning rate.
    lr_scheduler_gen: str = None  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    # lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})
    lr_scheduler_disc: str = None  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    # lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.5, 0.9], "weight_decay": 0.0})
    steps_to_start_discriminator: int = 200000

    def __post_init__(self):
        super().__post_init__()
        self.generator_model_params["cond_channels"] = self.audio.num_mels
