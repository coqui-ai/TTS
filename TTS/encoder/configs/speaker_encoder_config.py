from dataclasses import asdict, dataclass

from TTS.encoder.configs.base_encoder_config import BaseEncoderConfig


@dataclass
class SpeakerEncoderConfig(BaseEncoderConfig):
    """Defines parameters for Speaker Encoder model."""

    model: str = "speaker_encoder"
    class_name_key: str = "speaker_name"
