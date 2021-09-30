from dataclasses import dataclass, field
from typing import Dict, List

from TTS.stt.configs.shared_configs import BaseSTTConfig
from TTS.stt.models.deep_speech import DeepSpeechArgs


@dataclass
class DeepSpeechConfig(BaseSTTConfig):
    """Defines parameters for VITS End2End TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (VitsArgs):
            Model architecture arguments. Defaults to `VitsArgs()`.

        grad_clip (List):
            Gradient clipping thresholds for each optimizer. Defaults to `[5.0, 5.0]`.

        lr (float):
            Initial learning rate. Defaults to 0.0002.

        lr_scheduler_gen (str):
            Name of the learning rate scheduler for the generator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_gen_params (dict):
            Parameters for the learning rate scheduler of the generator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`.

        lr_scheduler_disc (str):
            Name of the learning rate scheduler for the discriminator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_disc_params (dict):
            Parameters for the learning rate scheduler of the discriminator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`.

        scheduler_after_epoch (bool):
            If true, step the schedulers after each epoch else after each step. Defaults to `False`.

        optimizer (str):
            Name of the optimizer to use with both the generator and the discriminator networks. One of the
            `torch.optim.*`. Defaults to `AdamW`.

        kl_loss_alpha (float):
            Loss weight for KL loss. Defaults to 1.0.

        disc_loss_alpha (float):
            Loss weight for the discriminator loss. Defaults to 1.0.

        gen_loss_alpha (float):
            Loss weight for the generator loss. Defaults to 1.0.

        feat_loss_alpha (float):
            Loss weight for the feature matching loss. Defaults to 1.0.

        mel_loss_alpha (float):
            Loss weight for the mel loss. Defaults to 45.0.

        return_wav (bool):
            If true, data loader returns the waveform as well as the other outputs. Do not change. Defaults to `True`.

        compute_linear_spec (bool):
            If true, the linear spectrogram is computed and returned alongside the mel output. Do not change. Defaults to `True`.

        sort_by_audio_len (bool):
            If true, dataloder sorts the data by audio length else sorts by the input text length. Defaults to `True`.

        min_seq_len (int):
            Minimum sequnce length to be considered for training. Defaults to `0`.

        max_seq_len (int):
            Maximum sequnce length to be considered for training. Defaults to `500000`.

        r (int):
            Number of spectrogram frames to be generated at a time. Do not change. Defaults to `1`.

        add_blank (bool):
            If true, a blank token is added in between every character. Defaults to `True`.

        test_sentences (List[str]):
            List of sentences to be used for testing.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.stt.configs import DeepSpeechConfig
        >>> config = DeepSpeechConfig()
    """

    model: str = "deep_speech"
    # model specific params
    model_args: DeepSpeechArgs = field(default_factory=DeepSpeechArgs)

    # optimizer
    grad_clip: float = 10
    lr: float = 0.0001
    lr_scheduler: str = "ExponentialLR"
    lr_scheduler_params: Dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: Dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # overrides
    loss_masking: bool = True
    feature_extractor: str = "MFCC"
    sort_by_audio_len: bool = True
    min_seq_len: int = 0
    max_seq_len: int = 500000
