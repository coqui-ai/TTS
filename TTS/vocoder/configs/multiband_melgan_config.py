from dataclasses import asdict, dataclass, field

from .shared_configs import BaseGANVocoderConfig


@dataclass
class MultibandMelganConfig(BaseGANVocoderConfig):
    """Defines parameters for MultiBandMelGAN vocoder."""
    model: str = "multiband_melgan"

    # Model specific params
    discriminator_model: str = "melgan_multiscale_discriminator"
    discriminator_model_params: dict = field(
        default_factory=lambda: {
            "base_channels": 16,
            "max_channels": 512,
            "downsample_factors": [4, 4, 4]
        })
    generator_model: str = "multiband_melgan_generator"
    generator_model_params: dict = field(
        default_factory=lambda: {
            "upsample_factors": [8, 4, 2],
            "num_res_blocks": 4
        })
    use_pqmf: bool = True

    # optimizer - overrides
    lr_gen: float = 0.0001  # Initial learning rate.
    lr_disc: float = 0.0001  # Initial learning rate.
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {
        "betas": [0.8, 0.99],
        "weight_decay": 0.0
    })
    lr_scheduler_gen: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {
        "gamma": 0.5,
        "milestones": [100000, 200000, 300000, 400000, 500000, 600000]
    })
    lr_scheduler_disc: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {
        "gamma": 0.5,
        "milestones": [100000, 200000, 300000, 400000, 500000, 600000]
    })

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
        default_factory=lambda: {
            "n_ffts": [384, 683, 171],
            "hop_lengths": [30, 60, 10],
            "win_lengths": [150, 300, 60]
        })

    # loss weights - overrides
    stft_loss_weight: float = 0.5
    subband_stft_loss_weight: float = 0
    mse_G_loss_weight: float = 2.5
    hinge_G_loss_weight: float = 0
    feat_match_loss_weight: float = 108
    l1_spec_loss_weight: float = 0

    # optimizer parameters
    lr: float = 1e-4
    wd: float = 1e-6
