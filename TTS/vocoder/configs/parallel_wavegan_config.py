from dataclasses import dataclass, field

from .shared_configs import BaseGANVocoderConfig


@dataclass
class ParallelWaveganConfig(BaseGANVocoderConfig):
    """Defines parameters for ParallelWavegan vocoder.

    Args:
        model (str):
            Model name used for selecting the right configuration at initialization. Defaults to `gan`.
        discriminator_model (str): One of the discriminators from `TTS.vocoder.models.*_discriminator`. Defaults to
            'parallel_wavegan_discriminator`.
        discriminator_model_params (dict): The discriminator model kwargs. Defaults to
            '{"num_layers": 10}`
        generator_model (str): One of the generators from TTS.vocoder.models.*`. Every other non-GAN vocoder model is
            considered as a generator too. Defaults to `parallel_wavegan_generator`.
        generator_model_param (dict):
            The generator model kwargs. Defaults to `{"upsample_factors": [4, 4, 4, 4], "stacks": 3, "num_res_blocks": 30}`.
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
            Feature matching loss weight that multiplies the computed loss before summing up the total loss. faults to 0.
        l1_spec_loss_weight (float):
            L1 spectrogram loss weight that multiplies the computed loss before summing up the total loss. Defaults to 0.
        lr_gen (float):
            Generator model initial learning rate. Defaults to 0.0002.
        lr_disc (float):
            Discriminator model initial learning rate. Defaults to 0.0002.
        optimizer (torch.optim.Optimizer):
            Optimizer used for the training. Defaults to `AdamW`.
        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`
        lr_scheduler_gen (torch.optim.Scheduler):
            Learning rate scheduler for the generator. Defaults to `ExponentialLR`.
        lr_scheduler_gen_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"gamma": 0.5, "step_size": 200000, "last_epoch": -1}`.
        lr_scheduler_disc (torch.optim.Scheduler):
            Learning rate scheduler for the discriminator. Defaults to `ExponentialLR`.
        lr_scheduler_dict_params (dict):
            Parameters for the discriminator learning rate scheduler. Defaults to `{"gamma": 0.5, "step_size": 200000, "last_epoch": -1}`.
    """

    model: str = "parallel_wavegan"

    # Model specific params
    discriminator_model: str = "parallel_wavegan_discriminator"
    discriminator_model_params: dict = field(default_factory=lambda: {"num_layers": 10})
    generator_model: str = "parallel_wavegan_generator"
    generator_model_params: dict = field(
        default_factory=lambda: {"upsample_factors": [4, 4, 4, 4], "stacks": 3, "num_res_blocks": 30}
    )

    # Training - overrides
    batch_size: int = 6
    seq_len: int = 25600
    pad_short: int = 2000
    use_noise_augment: bool = False
    use_cache: bool = True
    steps_to_start_discriminator: int = 200000
    target_loss: str = "loss_1"

    # LOSS PARAMETERS - overrides
    use_stft_loss: bool = True
    use_subband_stft_loss: bool = False
    use_mse_gan_loss: bool = True
    use_hinge_gan_loss: bool = False
    use_feat_match_loss: bool = False  # requires MelGAN Discriminators (MelGAN and HifiGAN)
    use_l1_spec_loss: bool = False

    stft_loss_params: dict = field(
        default_factory=lambda: {
            "n_ffts": [1024, 2048, 512],
            "hop_lengths": [120, 240, 50],
            "win_lengths": [600, 1200, 240],
        }
    )

    # loss weights - overrides
    stft_loss_weight: float = 0.5
    subband_stft_loss_weight: float = 0
    mse_G_loss_weight: float = 2.5
    hinge_G_loss_weight: float = 0
    feat_match_loss_weight: float = 0
    l1_spec_loss_weight: float = 0

    # optimizer overrides
    lr_gen: float = 0.0002  # Initial learning rate.
    lr_disc: float = 0.0002  # Initial learning rate.
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "weight_decay": 0.0})
    lr_scheduler_gen: str = "StepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.5, "step_size": 200000, "last_epoch": -1})
    lr_scheduler_disc: str = "StepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(
        default_factory=lambda: {"gamma": 0.5, "step_size": 200000, "last_epoch": -1}
    )
    scheduler_after_epoch: bool = False
