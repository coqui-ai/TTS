from dataclasses import asdict, dataclass, field

from .shared_configs import BaseGANVocoderConfig


@dataclass
class FullbandMelganConfig(BaseGANVocoderConfig):
    """Defines parameters for FullbandMelGAN vocoder."""
    model: str = "melgan"

    # Model specific params
    discriminator_model: str = "melgan_multiscale_discriminator"
    discriminator_model_params: dict = field(
        default_factory=lambda: {
            "base_channels": 16,
            "max_channels": 512,
            "downsample_factors": [4, 4, 4]
        })
    generator_model: str = "melgan_generator"
    generator_model_params: dict = field(
        default_factory=lambda: {
            "upsample_factors": [8, 8, 2, 2],
            "num_res_blocks": 4
        })

    # Training - overrides
    batch_size: int = 16
    seq_len: int = 8192
    pad_short: int = 2000
    use_noise_augment: bool = True
    use_cache: bool = True

    # LOSS PARAMETERS - overrides
    use_stft_loss: bool = True
    use_subband_stft_loss: bool = False
    use_mse_gan_loss: bool = True
    use_hinge_gan_loss: bool = False
    use_feat_match_loss: bool = True  # requires MelGAN Discriminators (MelGAN and HifiGAN)
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
    feat_match_loss_weight: float = 108
    l1_spec_loss_weight: float = 0
