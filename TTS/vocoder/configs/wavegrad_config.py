from dataclasses import dataclass, field

from TTS.vocoder.configs.shared_configs import BaseVocoderConfig
from TTS.vocoder.models.wavegrad import WavegradArgs


@dataclass
class WavegradConfig(BaseVocoderConfig):
    """Defines parameters for WaveGrad vocoder.
    Example:

        >>> from TTS.vocoder.configs import WavegradConfig
        >>> config = WavegradConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `wavegrad`.
        generator_model (str): One of the generators from TTS.vocoder.models.*`. Every other non-GAN vocoder model is
            considered as a generator too. Defaults to `wavegrad`.
        model_params (WavegradArgs): Model parameters. Check `WavegradArgs` for default values.
        target_loss (str):
            Target loss name that defines the quality of the model. Defaults to `avg_wavegrad_loss`.
        epochs (int):
            Number of epochs to traing the model. Defaults to 10000.
        batch_size (int):
            Batch size used at training. Larger values use more memory. Defaults to 96.
        seq_len (int):
            Audio segment length used at training. Larger values use more memory. Defaults to 6144.
        use_cache (bool):
            enable / disable in memory caching of the computed features. It can cause OOM error if the system RAM is
            not large enough. Defaults to True.
        mixed_precision (bool):
            enable / disable mixed precision training. Default is True.
        eval_split_size (int):
            Number of samples used for evalutaion. Defaults to 50.
        train_noise_schedule (dict):
            Training noise schedule. Defaults to
            `{"min_val": 1e-6, "max_val": 1e-2, "num_steps": 1000}`
        test_noise_schedule (dict):
            Inference noise schedule. For a better performance, you may need to use `bin/tune_wavegrad.py` to find a
            better schedule. Defaults to
            `
            {
                "min_val": 1e-6,
                "max_val": 1e-2,
                "num_steps": 50,
            }
            `
        grad_clip (float):
            Gradient clipping threshold. If <= 0.0, no clipping is applied. Defaults to 1.0
        lr (float):
            Initila leraning rate. Defaults to 1e-4.
        lr_scheduler (str):
            One of the learning rate schedulers from `torch.optim.scheduler.*`. Defaults to `MultiStepLR`.
        lr_scheduler_params (dict):
            kwargs for the scheduler. Defaults to `{"gamma": 0.5, "milestones": [100000, 200000, 300000, 400000, 500000, 600000]}`
    """

    model: str = "wavegrad"
    # Model specific params
    generator_model: str = "wavegrad"
    model_params: WavegradArgs = field(default_factory=WavegradArgs)
    target_loss: str = "loss"  # loss value to pick the best model to save after each epoch

    # Training - overrides
    epochs: int = 10000
    batch_size: int = 96
    seq_len: int = 6144
    use_cache: bool = True
    mixed_precision: bool = True
    eval_split_size: int = 50

    # NOISE SCHEDULE PARAMS
    train_noise_schedule: dict = field(default_factory=lambda: {"min_val": 1e-6, "max_val": 1e-2, "num_steps": 1000})

    test_noise_schedule: dict = field(
        default_factory=lambda: {  # inference noise schedule. Try TTS/bin/tune_wavegrad.py to find the optimal values.
            "min_val": 1e-6,
            "max_val": 1e-2,
            "num_steps": 50,
        }
    )

    # optimizer overrides
    grad_clip: float = 1.0
    lr: float = 1e-4  # Initial learning rate.
    lr_scheduler: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_params: dict = field(
        default_factory=lambda: {"gamma": 0.5, "milestones": [100000, 200000, 300000, 400000, 500000, 600000]}
    )
