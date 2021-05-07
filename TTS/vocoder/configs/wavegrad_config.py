from dataclasses import asdict, dataclass, field

from .shared_configs import BaseVocoderConfig


@dataclass
class WavegradConfig(BaseVocoderConfig):
    """Defines parameters for Wavernn vocoder."""
    model: str = 'wavegrad'
    # Model specific params
    generator_model: str = "wavegrad"
    model_params: dict = field(
        default_factory=lambda: {
            "use_weight_norm":
            True,
            "y_conv_channels":
            32,
            "x_conv_channels":
            768,
            "ublock_out_channels": [512, 512, 256, 128, 128],
            "dblock_out_channels": [128, 128, 256, 512],
            "upsample_factors": [4, 4, 4, 2, 2],
            "upsample_dilations": [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8],
                                   [1, 2, 4, 8], [1, 2, 4, 8]]
        })
    target_loss: str = 'avg_wavegrad_loss'  # loss value to pick the best model to save after each epoch

    # Training - overrides
    epochs: int = 10000
    batch_size: int = 96
    seq_len: int = 6144
    use_cache: bool = True
    steps_to_start_discriminator: int = 200000
    mixed_precision: bool = True
    eval_split_size: int = 50

    # NOISE SCHEDULE PARAMS
    train_noise_schedule: dict = field(default_factory=lambda: {
        "min_val": 1e-6,
        "max_val": 1e-2,
        "num_steps": 1000
    })

    test_noise_schedule: dict = field(default_factory=lambda: { # inference noise schedule. Try TTS/bin/tune_wavegrad.py to find the optimal values.
        "min_val": 1e-6,
        "max_val": 1e-2,
        "num_steps": 50
    })

    # optimizer overrides
    grad_clip: float = 1.0
    lr: float = 1e-4  # Initial learning rate.
    lr_scheduler: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_params: dict = field(
        default_factory=lambda: {
            "gamma": 0.5,
            "milestones": [100000, 200000, 300000, 400000, 500000, 600000]
        })
