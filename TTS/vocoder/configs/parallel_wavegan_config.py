from dataclasses import asdict, dataclass, field

from .shared_configs import BaseGANVocoderConfig


@dataclass
class ParallelWaveganConfig(BaseGANVocoderConfig):
    """Defines parameters for ParallelWavegan vocoder."""
    model: str = "parallel_wavegan"

    # Model specific params
    discriminator_model: str = "parallel_wavegan_discriminator"
    discriminator_model_params: dict = field(
        default_factory=lambda: {
            "num_layers": 10
        })
    generator_model: str = "parallel_wavegan_generator"
    generator_model_params: dict = field(
        default_factory=lambda: {
            "upsample_factors":[4, 4, 4, 4],
            "stacks": 3,
            "num_res_blocks": 30
        })

    # Training - overrides
    batch_size: int = 6
    seq_len: int = 25600
    pad_short: int = 2000
    use_noise_augment: bool = False
    use_cache: bool = True
    steps_to_start_discriminator: int = 200000

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
            "win_lengths": [600, 1200, 240]
        })

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
    optimizer_params: dict = field(default_factory=lambda: {
        "betas": [0.8, 0.99],
        "weight_decay": 0.0
    })
    lr_scheduler_gen: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {
        "gamma": 0.999,
        "last_epoch": -1
    })
    lr_scheduler_disc: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {
        "gamma": 0.999,
        "last_epoch": -1
    })
