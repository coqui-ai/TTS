from dataclasses import dataclass, field

from TTS.config import BaseAudioConfig, BaseTrainingConfig


@dataclass
class BaseVocoderConfig(BaseTrainingConfig):
    """Shared parameters among all the vocoder models.
    Args:
        audio (BaseAudioConfig):
            Audio processor config instance. Defaultsto `BaseAudioConfig()`.
        use_noise_augment (bool):
            Augment the input audio with random noise. Defaults to False/
        eval_split_size (int):
            Number of instances used for evaluation. Defaults to 10.
        data_path (str):
            Root path of the training data. All the audio files found recursively from this root path are used for
            training. Defaults to `""`.
        feature_path (str):
            Root path to the precomputed feature files. Defaults to None.
        seq_len (int):
            Length of the waveform segments used for training. Defaults to 1000.
        pad_short (int):
            Extra padding for the waveforms shorter than `seq_len`. Defaults to 0.
        conv_path (int):
            Extra padding for the feature frames against convolution of the edge frames. Defaults to MISSING.
            Defaults to 0.
        use_cache (bool):
            enable / disable in memory caching of the computed features. If the RAM is not enough, if may cause OOM.
            Defaults to False.
        epochs (int):
            Number of training epochs to. Defaults to 10000.
        wd (float):
            Weight decay.
         optimizer (torch.optim.Optimizer):
            Optimizer used for the training. Defaults to `AdamW`.
        optimizer_params (dict):
            Optimizer kwargs. Defaults to `{"betas": [0.8, 0.99], "weight_decay": 0.0}`
    """

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # dataloading
    use_noise_augment: bool = False  # enable/disable random noise augmentation in spectrograms.
    eval_split_size: int = 10  # number of samples used for evaluation.
    # dataset
    data_path: str = ""  # root data path. It finds all wav files recursively from there.
    feature_path: str = None  # if you use precomputed features
    seq_len: int = 1000  # signal length used in training.
    pad_short: int = 0  # additional padding for short wavs
    conv_pad: int = 0  # additional padding against convolutions applied to spectrograms
    use_cache: bool = False  # use in memory cache to keep the computed features. This might cause OOM.
    # OPTIMIZER
    epochs: int = 10000  # total number of epochs to train.
    wd: float = 0.0  # Weight decay weight.
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "weight_decay": 0.0})


@dataclass
class BaseGANVocoderConfig(BaseVocoderConfig):
    """Base config class used among all the GAN based vocoders.
    Args:
        use_stft_loss (bool):
            enable / disable the use of STFT loss. Defaults to True.
        use_subband_stft_loss (bool):
            enable / disable the use of Subband STFT loss. Defaults to True.
        use_mse_gan_loss (bool):
            enable / disable the use of Mean Squared Error based GAN loss. Defaults to True.
        use_hinge_gan_loss (bool):
            enable / disable the use of Hinge GAN loss. Defaults to True.
        use_feat_match_loss (bool):
            enable / disable feature matching loss. Defaults to True.
        use_l1_spec_loss (bool):
            enable / disable L1 spectrogram loss. Defaults to True.
        stft_loss_weight (float):
            Loss weight that multiplies the computed loss value. Defaults to 0.
        subband_stft_loss_weight (float):
            Loss weight that multiplies the computed loss value. Defaults to 0.
        mse_G_loss_weight (float):
            Loss weight that multiplies the computed loss value. Defaults to 1.
        hinge_G_loss_weight (float):
            Loss weight that multiplies the computed loss value. Defaults to 0.
        feat_match_loss_weight (float):
            Loss weight that multiplies the computed loss value. Defaults to 100.
        l1_spec_loss_weight (float):
            Loss weight that multiplies the computed loss value. Defaults to 45.
        stft_loss_params (dict):
            Parameters for the STFT loss. Defaults to `{"n_ffts": [1024, 2048, 512], "hop_lengths": [120, 240, 50], "win_lengths": [600, 1200, 240]}`.
        l1_spec_loss_params (dict):
            Parameters for the L1 spectrogram loss. Defaults to
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
        target_loss (str):
            Target loss name that defines the quality of the model. Defaults to `G_avg_loss`.
        grad_clip (list):
            A list of gradient clipping theresholds for each optimizer. Any value less than 0 disables clipping.
            Defaults to [5, 5].
        lr_gen (float):
            Generator model initial learning rate. Defaults to 0.0002.
        lr_disc (float):
            Discriminator model initial learning rate. Defaults to 0.0002.
        lr_scheduler_gen (torch.optim.Scheduler):
            Learning rate scheduler for the generator. Defaults to `ExponentialLR`.
        lr_scheduler_gen_params (dict):
            Parameters for the generator learning rate scheduler. Defaults to `{"gamma": 0.999, "last_epoch": -1}`.
        lr_scheduler_disc (torch.optim.Scheduler):
            Learning rate scheduler for the discriminator. Defaults to `ExponentialLR`.
        lr_scheduler_disc_params (dict):
            Parameters for the discriminator learning rate scheduler. Defaults to `{"gamma": 0.999, "last_epoch": -1}`.
        scheduler_after_epoch (bool):
            Whether to update the learning rate schedulers after each epoch. Defaults to True.
        use_pqmf (bool):
            enable / disable PQMF for subband approximation at training. Defaults to False.
        steps_to_start_discriminator (int):
            Number of steps required to start training the discriminator. Defaults to 0.
        diff_samples_for_G_and_D (bool):
            enable / disable use of different training samples for the generator and the discriminator iterations.
            Enabling it results in slower iterations but faster convergance in some cases. Defaults to False.
    """

    model: str = "gan"

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
    feat_match_loss_weight: float = 100
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

    target_loss: str = "loss_0"  # loss value to pick the best model to save after each epoch

    # optimizer
    grad_clip: float = field(default_factory=lambda: [5, 5])
    lr_gen: float = 0.0002  # Initial learning rate.
    lr_disc: float = 0.0002  # Initial learning rate.
    lr_scheduler_gen: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999, "last_epoch": -1})
    scheduler_after_epoch: bool = True

    use_pqmf: bool = False  # enable/disable using pqmf for multi-band training. (Multi-band MelGAN)
    steps_to_start_discriminator = 0  # start training the discriminator after this number of steps.
    diff_samples_for_G_and_D: bool = False  # use different samples for G and D training steps.
