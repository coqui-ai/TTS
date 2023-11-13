from dataclasses import dataclass, field
from typing import List

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.xtts import XttsArgs, XttsAudioConfig


@dataclass
class XttsConfig(BaseTTSConfig):
    """Defines parameters for XTTS TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (XttsArgs):
            Model architecture arguments. Defaults to `XttsArgs()`.

        audio (XttsAudioConfig):
            Audio processing configuration. Defaults to `XttsAudioConfig()`.

        model_dir (str):
            Path to the folder that has all the XTTS models. Defaults to None.

        temperature (float):
            Temperature for the autoregressive model inference. Larger values makes predictions more creative sacrificing stability. Defaults to `0.2`.

        length_penalty (float):
            Exponential penalty to the length that is used with beam-based generation. It is applied as an exponent to the sequence length,
            which in turn is used to divide the score of the sequence. Since the score is the log likelihood of the sequence (i.e. negative),
            length_penalty > 0.0 promotes longer sequences, while length_penalty < 0.0 encourages shorter sequences.

        repetition_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        num_gpt_outputs (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As XTTS is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        gpt_cond_len (int):
            Secs audio to be used as conditioning for the autoregressive model. Defaults to `12`.

        gpt_cond_chunk_len (int):
            Audio chunk size in secs. Audio is split into chunks and latents are extracted for each chunk. Then the
            latents are averaged. Chunking improves the stability. It must be <= gpt_cond_len.
            If gpt_cond_len == gpt_cond_chunk_len, no chunking. Defaults to `4`.

        max_ref_len (int):
            Maximum number of seconds of audio to be used as conditioning for the decoder. Defaults to `10`.

        sound_norm_refs (bool):
            Whether to normalize the conditioning audio. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.xtts_config import XttsConfig
        >>> config = XttsConfig()
    """

    model: str = "xtts"
    # model specific params
    model_args: XttsArgs = field(default_factory=XttsArgs)
    audio: XttsAudioConfig = field(default_factory=XttsAudioConfig)
    model_dir: str = None
    languages: List[str] = field(
        default_factory=lambda: [
            "en",
            "es",
            "fr",
            "de",
            "it",
            "pt",
            "pl",
            "tr",
            "ru",
            "nl",
            "cs",
            "ar",
            "zh-cn",
            "hu",
            "ko",
            "ja",
        ]
    )

    # inference params
    temperature: float = 0.85
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.85
    num_gpt_outputs: int = 1

    # cloning
    gpt_cond_len: int = 12
    gpt_cond_chunk_len: int = 4
    max_ref_len: int = 10
    sound_norm_refs: bool = False
