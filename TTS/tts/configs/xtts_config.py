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

        reperation_penalty (float):
            The parameter for repetition penalty. 1.0 means no penalty. Defaults to `2.0`.

        top_p (float):
            If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
            Defaults to `0.8`.

        cond_free_k (float):
            Knob that determines how to balance the conditioning free signal with the conditioning-present signal. [0,inf].
            As cond_free_k increases, the output becomes dominated by the conditioning-free signal.
            Formula is: output=cond_present_output*(cond_free_k+1)-cond_absenct_output*cond_free_k. Defaults to `2.0`.

        diffusion_temperature (float):
            Controls the variance of the noise fed into the diffusion model. [0,1]. Values at 0
            are the "mean" prediction of the diffusion network and will sound bland and smeared.
            Defaults to `1.0`.

        num_gpt_outputs (int):
            Number of samples taken from the autoregressive model, all of which are filtered using CLVP.
            As XTTS is a probabilistic model, more samples means a higher probability of creating something "great".
            Defaults to `16`.

        decoder_iterations (int):
            Number of diffusion steps to perform. [0,4000]. More steps means the network has more chances to iteratively refine
            the output, which should theoretically mean a higher quality output. Generally a value above 250 is not noticeably better,
            however. Defaults to `30`.

        decoder_sampler (str):
            Diffusion sampler to be used. `ddim` or `dpm++2m`. Defaults to `ddim`.
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
        default_factory=lambda: ["en", "es", "fr", "de", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn"]
    )

    # inference params
    temperature: float = 0.2
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_k: int = 50
    top_p: float = 0.8
    cond_free_k: float = 2.0
    diffusion_temperature: float = 1.0
    num_gpt_outputs: int = 16
    decoder_iterations: int = 30
    decoder_sampler: str = "ddim"
