from dataclasses import dataclass, field

from coqpit import MISSING

from TTS.config import BaseAudioConfig, BaseTrainingConfig


@dataclass
class BaseVocoderConfig(BaseTrainingConfig):
    """Shared parameters among all the vocoder models."""

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # dataloading
    use_noise_augment: bool = False  # enable/disable random noise augmentation in spectrograms.
    eval_split_size: int = 10  # number of samples used for evaluation.
    # dataset
    data_path: str = MISSING  # root data path. It finds all wav files recursively from there.
    feature_path: str = None  # if you use precomputed features
    seq_len: int = MISSING  # signal length used in training.
    pad_short: int = 0  # additional padding for short wavs
    conv_pad: int = 0  # additional padding against convolutions applied to spectrograms
    use_noise_augment: bool = False  # add noise to the audio signal for augmentation
    use_cache: bool = False  # use in memory cache to keep the computed features. This might cause OOM.
    # OPTIMIZER
    epochs: int = 10000  # total number of epochs to train.
    wd: float = 0.0  # Weight decay weight.


@dataclass
class BaseGANVocoderConfig(BaseVocoderConfig):
    """Common config interface for all the GAN based vocoder models."""

    # LOSS PARAMETERS
    use_stft_loss: bool = True
    use_subband_stft_loss: bool = True
    use_mse_gan_loss: bool = True
    use_hinge_gan_loss: bool = True
    use_feat_match_loss: bool = True  # requires MelGAN Discriminators (MelGAN and HifiGAN)
    use_l1_spec_loss: bool = True

    # loss weights
    stft_loss_weight: float = 0
    subband_stft_loss_weight: float = 0
    mse_G_loss_weight: float = 1
    hinge_G_loss_weight: float = 0
    feat_match_loss_weight: float = 10
    l1_spec_loss_weight: float = 45

    stft_loss_params: dict = field(
        default_factory=lambda: {
            "n_ffts": [1024, 2048, 512],
            "hop_lengths": [120, 240, 50],
            "win_lengths": [600, 1200, 240],
        }
    )

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

    target_loss: str = "avg_G_loss"  # loss value to pick the best model to save after each epoch

    # optimizer
    gen_clip_grad: float = -1  # Generator gradient clipping threshold. Apply gradient clipping if > 0
    disc_clip_grad: float = -1  # Discriminator gradient clipping threshold.
    lr_gen: float = 0.0002  # Initial learning rate.
    lr_disc: float = 0.0002  # Initial learning rate.
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "weight_decay": 0.0})
    lr_scheduler_gen: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})

    use_pqmf: bool = False  # enable/disable using pqmf for multi-band training. (Multi-band MelGAN)
    steps_to_start_discriminator = 0  # start training the discriminator after this number of steps.
    diff_samples_for_G_and_D: bool = False  # use different samples for G and D training steps.
