from dataclasses import dataclass

from TTS.tts.configs.tacotron_config import TacotronConfig


@dataclass
class Tacotron2Config(TacotronConfig):
    """Defines parameters for Tacotron2 based models."""

    model: str = "tacotron2"
