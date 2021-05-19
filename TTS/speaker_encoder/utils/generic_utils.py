import re

from TTS.speaker_encoder.model import SpeakerEncoder


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(c):
    model = SpeakerEncoder(
        c.model_params["input_dim"],
        c.model_params["proj_dim"],
        c.model_params["lstm_dim"],
        c.model_params["num_lstm_layers"],
    )
    return model
