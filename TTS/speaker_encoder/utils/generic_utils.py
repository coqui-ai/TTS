import re

from TTS.speaker_encoder.model import SpeakerEncoder


def to_camel(text):
    text = text.capitalize()
    return re.sub(r"(?!^)_([a-zA-Z])", lambda m: m.group(1).upper(), text)


def setup_model(c):
    model = SpeakerEncoder(c.model["input_dim"], c.model["proj_dim"],
                           c.model["lstm_dim"], c.model["num_lstm_layers"])
    return model