from dataclasses import asdict, dataclass, field

from .shared_configs import BaseVocoderConfig


@dataclass
class WavernnConfig(BaseVocoderConfig):
    """Defines parameters for Wavernn vocoder."""

    model: str = "wavernn"

    # Model specific params
    mode: str = "mold"  # mold [string], gauss [string], bits [int]
    mulaw: bool = True  # apply mulaw if mode is bits
    generator_model: str = "WaveRNN"
    wavernn_model_params: dict = field(
        default_factory=lambda: {
            "rnn_dims": 512,
            "fc_dims": 512,
            "compute_dims": 128,
            "res_out_dims": 128,
            "num_res_blocks": 10,
            "use_aux_net": True,
            "use_upsample_net": True,
            "upsample_factors": [4, 8, 8],  # this needs to correctly factorise hop_length
        }
    )

    # Inference
    batched: bool = True
    target_samples: int = 11000
    overlap_samples: int = 550

    # Training - overrides
    epochs: int = 10000
    batch_size: int = 256
    seq_len: int = 1280
    padding: int = 2
    use_noise_augment: bool = False
    use_cache: bool = True
    steps_to_start_discriminator: int = 200000
    mixed_precision: bool = True
    eval_split_size: int = 50
    test_every_epochs: int = 10  # number of epochs to wait until the next test run (synthesizing a full audio clip).

    # optimizer overrides
    grad_clip: float = 4.0
    lr: float = 1e-4  # Initial learning rate.
    lr_scheduler: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_params: dict = field(default_factory=lambda: {"gamma": 0.5, "milestones": [200000, 400000, 600000]})
