from dataclasses import asdict, dataclass, field

from .shared_configs import BaseGANVocoderConfig


@dataclass
class HifiganConfig(BaseGANVocoderConfig):
    """Defines parameters for HifiGAN vocoder."""

    model: str = "hifigan"
    # model specific params
    discriminator_model: str = "hifigan_discriminator"
    generator_model: str = "hifigan_generator"
    generator_model_params: dict = field(
        default_factory=lambda: {
            "upsample_factors": [8, 8, 2, 2],
            "upsample_kernel_sizes": [16, 16, 4, 4],
            "upsample_initial_channel": 512,
            "resblock_kernel_sizes": [3, 7, 11],
            "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            "resblock_type": "1",
        }
    )

    # LOSS PARAMETERS - overrides
    use_stft_loss: bool = False
    use_subband_stft_loss: bool = False
    use_mse_gan_loss: bool = True
    use_hinge_gan_loss: bool = False
    use_feat_match_loss: bool = True  # requires MelGAN Discriminators (MelGAN and HifiGAN)
    use_l1_spec_loss: bool = True

    # loss weights - overrides
    stft_loss_weight: float = 0
    subband_stft_loss_weight: float = 0
    mse_G_loss_weight: float = 1
    hinge_G_loss_weight: float = 0
    feat_match_loss_weight: float = 108
    l1_spec_loss_weight: float = 45
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
    lr: float = 1e-4
    wd: float = 1e-6
