from dataclasses import dataclass, field

from TTS.vocoder.configs.shared_configs import BaseGANVocoderConfig


@dataclass
class MultibandMelganConfig(BaseGANVocoderConfig):
    """Defines parameters for MultiBandMelGAN vocoder.

    Example:

        >>> from TTS.vocoder.configs import MultibandMelganConfig
        >>> config = MultibandMelganConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `multiband_melgan`.
        discriminator_model (str): One of the discriminators from `TTS.vocoder.models.*_discriminator`. Defaults to
            'melgan_multiscale_discriminator`.
        discriminator_model_params (dict): The discriminator model parameters. Defaults to
            '{
                "base_channels": 16,
                "max_channels": 512,
                "downsample_factors": [4, 4, 4]
            }`
        generator_model (str): One of the generators from TTS.vocoder.models.*`. Every other non-GAN vocoder model is
            considered as a generator too. Defaults to `melgan_generator`.
        generator_model_param (dict):
            The generator model parameters. Defaults to `{"upsample_factors": [8, 4, 2], "num_res_blocks": 4}`.
        use_pqmf (bool):
            enable / disable PQMF modulation for multi-band training. Defaults to True.
        lr_gen (float):
            Initial learning rate for the generator model. Defaults to 0.0001.
        lr_disc (float):
            Initial learning rate for the discriminator model. Defaults to 0.0001.
        optimizer (torch.optim.Optimizer):
            Optimizer used for the training. Defaults to `AdamW`.
        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`
        lr_scheduler_gen (torch.optim.Scheduler):
            Learning rate scheduler for the generator. Defaults to `MultiStepLR`.
        lr_scheduler_gen_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to
            `{"gamma": 0.5, "milestones": [100000, 200000, 300000, 400000, 500000, 600000]}`.
        lr_scheduler_disc (torch.optim.Scheduler):
            Learning rate scheduler for the discriminator. Defaults to `MultiStepLR`.
        lr_scheduler_dict_params (dict):
            Parameters for the discriminator learning rate scheduler. Defaults to
            `{"gamma": 0.5, "milestones": [100000, 200000, 300000, 400000, 500000, 600000]}`.
        batch_size (int):
            Batch size used at training. Larger values use more memory. Defaults to 16.
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
        steps_to_start_discriminator (int):
            Number of steps required to start training the discriminator. Defaults to 0.
        use_stft_loss (bool):`
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
            enable / disable using L1 spectrogram loss originally used by HifiGAN model. Defaults to False.
        stft_loss_params (dict): STFT loss parameters. Default to
            `{"n_ffts": [1024, 2048, 512], "hop_lengths": [120, 240, 50], "win_lengths": [600, 1200, 240]}`
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

    model: str = "multiband_melgan"

    # Model specific params
    discriminator_model: str = "melgan_multiscale_discriminator"
    discriminator_model_params: dict = field(
        default_factory=lambda: {"base_channels": 16, "max_channels": 512, "downsample_factors": [4, 4, 4]}
    )
    generator_model: str = "multiband_melgan_generator"
    generator_model_params: dict = field(default_factory=lambda: {"upsample_factors": [8, 4, 2], "num_res_blocks": 4})
    use_pqmf: bool = True

    # optimizer - overrides
    lr_gen: float = 0.0001  # Initial learning rate.
    lr_disc: float = 0.0001  # Initial learning rate.
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "weight_decay": 0.0})
    lr_scheduler_gen: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params: dict = field(
        default_factory=lambda: {"gamma": 0.5, "milestones": [100000, 200000, 300000, 400000, 500000, 600000]}
    )
    lr_scheduler_disc: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(
        default_factory=lambda: {"gamma": 0.5, "milestones": [100000, 200000, 300000, 400000, 500000, 600000]}
    )

    # Training - overrides
    batch_size: int = 64
    seq_len: int = 16384
    pad_short: int = 2000
    use_noise_augment: bool = False
    use_cache: bool = True
    steps_to_start_discriminator: bool = 200000

    # LOSS PARAMETERS - overrides
    use_stft_loss: bool = True
    use_subband_stft_loss: bool = True
    use_mse_gan_loss: bool = True
    use_hinge_gan_loss: bool = False
    use_feat_match_loss: bool = False  # requires MelGAN Discriminators (MelGAN and HifiGAN)
    use_l1_spec_loss: bool = False

    subband_stft_loss_params: dict = field(
        default_factory=lambda: {"n_ffts": [384, 683, 171], "hop_lengths": [30, 60, 10], "win_lengths": [150, 300, 60]}
    )

    # loss weights - overrides
    stft_loss_weight: float = 0.5
    subband_stft_loss_weight: float = 0
    mse_G_loss_weight: float = 2.5
    hinge_G_loss_weight: float = 0
    feat_match_loss_weight: float = 108
    l1_spec_loss_weight: float = 0
