from TTS.stt.datasets.tokenizer import Tokenizer
from TTS.tts.utils.text.symbols import make_symbols, parse_symbols
from TTS.utils.generic_utils import find_module


def setup_stt_model(config):
    print(" > Using model: {}".format(config.model))
    # fetch the right model implementation.
    if "base_model" in config and config["base_model"] is not None:
        MyModel = find_module("TTS.stt.models", config.base_model.lower())
    else:
        MyModel = find_module("TTS.stt.models", config.model.lower())
    model = MyModel(config)
    return model
