from dataclasses import asdict, dataclass

from TTS.encoder.speaker_encoder_config import SpeakerEncoderConfig


@dataclass
class EmotionEncoderConfig(SpeakerEncoderConfig):
    """Defines parameters for Speaker Encoder model."""

    model: str = "emotion_encoder"

    def check_values(self):
        super().check_values()
        c = asdict(self)
        assert (
            c["model_params"]["input_dim"] == self.audio.num_mels
        ), " [!] model input dimendion must be equal to melspectrogram dimension."
