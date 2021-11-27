from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.vits import VitsArgs


@dataclass
class VitsConfig(BaseTTSConfig):
    """Defines parameters for VITS End2End TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (VitsArgs):
            Model architecture arguments. Defaults to `VitsArgs()`.

        grad_clip (List):
            Gradient clipping thresholds for each optimizer. Defaults to `[5.0, 5.0]`.

        lr_gen (float):
            Initial learning rate for the generator. Defaults to 0.0002.

        lr_disc (float):
            Initial learning rate for the discriminator. Defaults to 0.0002.

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

        test_sentences (List[List]):
            List of sentences with speaker and language information to be used for testing.

        language_ids_file (str):
            Path to the language ids file.

        use_language_embedding (bool):
            If true, language embedding is used. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> config = VitsConfig()
    """

    model: str = "vits"
    # model specific params
    model_args: VitsArgs = field(default_factory=VitsArgs)

    # optimizer
    grad_clip: List[float] = field(default_factory=lambda: [1000, 1000])
    lr_gen: float = 0.0002
    lr_disc: float = 0.0002
    lr_scheduler_gen: str = "ExponentialLR"
    lr_scheduler_gen_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    lr_scheduler_disc: str = "ExponentialLR"
    lr_scheduler_disc_params: dict = field(default_factory=lambda: {"gamma": 0.999875, "last_epoch": -1})
    scheduler_after_epoch: bool = True
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})

    # loss params
    kl_loss_alpha: float = 1.0
    disc_loss_alpha: float = 1.0
    gen_loss_alpha: float = 1.0
    feat_loss_alpha: float = 1.0
    mel_loss_alpha: float = 45.0
    dur_loss_alpha: float = 1.0
    speaker_encoder_loss_alpha: float = 1.0

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # overrides
    sort_by_audio_len: bool = True
    min_seq_len: int = 0
    max_seq_len: int = 500000
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # testing
    test_sentences: List[List] = field(
        default_factory=lambda: [
            ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent."],
            ["Be a voice, not an echo."],
            ["I'm sorry Dave. I'm afraid I can't do that."],
            ["This cake is great. It's so delicious and moist."],
            ["Prior to November 22, 1963."],
        ]
    )

    # multi-speaker settings
    # use speaker embedding layer
    num_speakers: int = 0
    use_speaker_embedding: bool = False
    speakers_file: str = None
    speaker_embedding_channels: int = 256
    language_ids_file: str = None
    use_language_embedding: bool = False

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: str = None
    d_vector_dim: int = None

    def __post_init__(self):
        for key, val in self.model_args.items():
            if hasattr(self, key):
                self[key] = val
