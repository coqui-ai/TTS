from dataclasses import dataclass, field

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.models.tortoise import TortoiseArgs, TortoiseAudioConfig


@dataclass
class TortoiseConfig(BaseTTSConfig):
    """Defines parameters for Tortoise TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (TortoiseArgs):
            Model architecture arguments. Defaults to `TortoiseArgs()`.

        audio (TortoiseAudioConfig):
            Audio processing configuration. Defaults to `TortoiseAudioConfig()`.
    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> config = VitsConfig()
    """

    model: str = "tortoise"
    # model specific params
    model_args: TortoiseArgs = field(default_factory=TortoiseArgs)
    audio: TortoiseAudioConfig = TortoiseAudioConfig()
    model_dir: str = None

    # settings
    temperature: float = 0.2
    length_penalty: float = 1.0
    repetition_penalty: float = 2.0
    top_p: float = 0.8
    cond_free_k: float = 2.0
    diffusion_temperature: float = 1.0

    # inference params
    num_autoregressive_samples: int = 16
    diffusion_iterations: int = 30
    sampler: str = "ddim"
