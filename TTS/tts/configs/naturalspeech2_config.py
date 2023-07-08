from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.naturalspeech2 import Naturalspeech2Args, Naturalspeech2AudioConfig


@dataclass
class Naturalspeech2Config(BaseTTSConfig):
    """Defines parameters for naturalspeech2 End2End TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (Naturalspeech2Args):
            Model architecture arguments. Defaults to `Naturalspeech2Args()`.

        audio (Naturalspeech2Args):
            Audio processing configuration. Defaults to `Naturalspeech2Args()`.

        grad_clip (List):
            Gradient clipping thresholds for each optimizer. Defaults to `[1000.0, 1000.0]`.

        lr (float):
            Initial learning rate for the generator. Defaults to 0.0002.
        lr_scheduler (str):
            Name of the learning rate scheduler for the generator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_params (dict):
            Parameters for the learning rate scheduler of the generator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`

        scheduler_after_epoch (bool):
            If true, step the schedulers after each epoch else after each step. Defaults to `False`.

        optimizer (str):
            Name of the optimizer to use with both the generator and the discriminator networks. One of the
            `torch.optim.*`. Defaults to `AdamW`.


        return_wav (bool):
            If true, data loader returns the waveform as well as the other outputs. Do not change. Defaults to `True`.

        compute_linear_spec (bool):
            If true, the linear spectrogram is computed and returned alongside the mel output. Do not change. Defaults to `True`.

        use_weighted_sampler (bool):
            If true, use weighted sampler with bucketing for balancing samples between datasets used in training. Defaults to `False`.

        weighted_sampler_attrs (dict):
            Key retuned by the formatter to be used for weighted sampler. For example `{"root_path": 2.0, "speaker_name": 1.0}` sets sample probabilities
            by overweighting `root_path` by 2.0. Defaults to `{}`.

        weighted_sampler_multipliers (dict):
            Weight each unique value of a key returned by the formatter for weighted sampling.
            For example `{"root_path":{"/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-100/":1.0, "/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-360/": 0.5}`.
            It will sample instances from `train-clean-100` 2 times more than `train-clean-360`. Defaults to `{}`.

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

        >>> from TTS.tts.configs.vits_config import Naturalspeech2Config
        >>> config = Naturalspeech2Config()
    """

    model: str = "naturalspeech2"
    # model specific params
    model_args: Naturalspeech2Args = field(default_factory=Naturalspeech2Args)
    audio: Naturalspeech2AudioConfig = Naturalspeech2AudioConfig()

    # optimizer
    lr_scheduler: str = "NoamLR"
    lr_scheduler_params: dict = field(default_factory=lambda: {"warmup_steps": 32000})
    # lr_scheduler: str = "ExponentialLR"
    # lr_scheduler_params: dict = field(default_factory=lambda: {"gamma": 0.98, "last_epoch": -1})
    lr: float = 1e-4
    scheduler_after_epoch: bool = False
    optimizer: str = "AdamW"
    optimizer_params: dict = field(default_factory=lambda: {"betas": [0.8, 0.99], "eps": 1e-9, "weight_decay": 0.01})
    grad_clip: float = 500.0

    # loss params
    data_loss_alpha: float = 1.0
    ce_loss_alpha: float = 0.01
    aligner_loss_alpha: float = 1.0
    duration_loss_alpha: float = 1.0
    pitch_loss_alpha: float = 1.0
    mel_loss_alpha: float = 0.0
    diffusion_loss_alpha: float = 1.0
    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # sampler params
    use_weighted_sampler: bool = False  # TODO: move it to the base config
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # dataset configs
    compute_f0: bool = True
    f0_cache_path: str = None
    use_voice_prompt: bool = True
    # testing
    test_sentences: List[List] = field(
        default_factory=lambda: [
            ["It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.", "/datasets/Final_mailabs_vctk/en_UK/wav24/elizabeth_klett_Female_en_UK/jane_eyre_01_f000007.wav"],
            ["Be a voice, not an echo, keep speaking and smiling.", "/datasets/Final_mailabs_vctk/en_UK/wav24/elizabeth_klett_Female_en_UK/jane_eyre_01_f000007.wav"],
            ["I'm sorry Dave. I'm afraid I can't do that.", "/datasets/en/libri+vctk/wav24/p281/p281_395.wav"],
            ["This cake is great. It's so delicious and moist.", "/datasets/en/libri+vctk/wav24/p281/p281_334.wav"],
            ["Prior to November 22, 1963.", "/datasets/en/libri+vctk/wav24/p281/p281_395.wav"],
        ]
    )