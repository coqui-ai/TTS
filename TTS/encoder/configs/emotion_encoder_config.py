from dataclasses import asdict, dataclass

from TTS.encoder.configs.base_encoder_config import BaseEncoderConfig


@dataclass
class EmotionEncoderConfig(BaseEncoderConfig):
    """Defines parameters for Emotion Encoder model."""

    model: str = "emotion_encoder"
    map_classid_to_classname: dict = None
    class_name_key: str = "emotion_name"
