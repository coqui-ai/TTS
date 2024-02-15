from dataclasses import dataclass, field

from TTS.tts.configs.shared_configs import BaseTTSConfig


@dataclass
class MatchaTTSConfig(BaseTTSConfig):
    model: str = "matcha_tts"
    num_chars: int = None
