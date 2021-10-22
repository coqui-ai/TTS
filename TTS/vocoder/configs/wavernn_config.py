from dataclasses import dataclass, field

from TTS.vocoder.configs.shared_configs import BaseVocoderConfig
from TTS.vocoder.models.wavernn import WavernnArgs


@dataclass
class WavernnConfig(BaseVocoderConfig):
    """Defines parameters for Wavernn vocoder.
    Example:

        >>> from TTS.vocoder.configs import WavernnConfig
        >>> config = WavernnConfig()

    Args:
        model (str):
            Model name used for selecting the right model at initialization. Defaults to `wavernn`.
        mode (str):
            Output mode of the WaveRNN vocoder. `mold` for Mixture of Logistic Distribution, `gauss` for a single
            Gaussian Distribution and `bits` for quantized bits as the model's output.
        mulaw (bool):
            enable / disable the use of Mulaw quantization for training. Only applicable if `mode == 'bits'`. Defaults
            to `True`.
        generator_model (str):
            One of the generators from TTS.vocoder.models.*`. Every other non-GAN vocoder model is
            considered as a generator too. Defaults to `WaveRNN`.
        wavernn_model_params (dict):
            kwargs for the WaveRNN model. Defaults to
            `{
                "rnn_dims": 512,
                "fc_dims": 512,
                "compute_dims": 128,
                "res_out_dims": 128,
                "num_res_blocks": 10,
                "use_aux_net": True,
                "use_upsample_net": True,
                "upsample_factors": [4, 8, 8]
            }`
        batched (bool):
            enable / disable the batched inference. It speeds up the inference by splitting the input into segments and
            processing the segments in a batch. Then it merges the outputs with a certain overlap and smoothing. If
            you set it False, without CUDA, it is too slow to be practical. Defaults to True.
        target_samples (int):
            Size of the segments in batched mode. Defaults to 11000.
        overlap_sampels (int):
            Size of the overlap between consecutive segments. Defaults to 550.
        batch_size (int):
            Batch size used at training. Larger values use more memory. Defaults to 256.
        seq_len (int):
            Audio segment length used at training. Larger values use more memory. Defaults to 1280.

        use_noise_augment (bool):
            enable / disable random noise added to the input waveform. The noise is added after computing the
            features. Defaults to True.
        use_cache (bool):
            enable / disable in memory caching of the computed features. It can cause OOM error if the system RAM is
            not large enough. Defaults to True.
        mixed_precision (bool):
            enable / disable mixed precision training. Default is True.
        eval_split_size (int):
            Number of samples used for evalutaion. Defaults to 50.
        num_epochs_before_test (int):
            Number of epochs waited to run the next evalution. Since inference takes some time, it is better to
            wait some number of epochs not ot waste training time. Defaults to 10.
        grad_clip (float):
            Gradient clipping threshold. If <= 0.0, no clipping is applied. Defaults to 4.0
        lr (float):
            Initila leraning rate. Defaults to 1e-4.
        lr_scheduler (str):
            One of the learning rate schedulers from `torch.optim.scheduler.*`. Defaults to `MultiStepLR`.
        lr_scheduler_params (dict):
            kwargs for the scheduler. Defaults to `{"gamma": 0.5, "milestones": [200000, 400000, 600000]}`
    """

    model: str = "wavernn"

    # Model specific params
    model_args: WavernnArgs = field(default_factory=WavernnArgs)
    target_loss: str = "loss"

    # Inference
    batched: bool = True
    target_samples: int = 11000
    overlap_samples: int = 550

    # Training - overrides
    epochs: int = 10000
    batch_size: int = 256
    seq_len: int = 1280
    use_noise_augment: bool = False
    use_cache: bool = True
    mixed_precision: bool = True
    eval_split_size: int = 50
    num_epochs_before_test: int = (
        10  # number of epochs to wait until the next test run (synthesizing a full audio clip).
    )

    # optimizer overrides
    grad_clip: float = 4.0
    lr: float = 1e-4  # Initial learning rate.
    lr_scheduler: str = "MultiStepLR"  # one of the schedulers from https:#pytorch.org/docs/stable/optim.html
    lr_scheduler_params: dict = field(default_factory=lambda: {"gamma": 0.5, "milestones": [200000, 400000, 600000]})
