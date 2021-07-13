from TTS.tts.utils.text.symbols import make_symbols, parse_symbols
from TTS.utils.generic_utils import find_module


def setup_model(config, data):
    print(" > Using model: {}".format(config.model))

    MyModel = find_module("TTS.tts.models", config.model.lower())
    # define set of characters used by the model
    if config.characters is not None:
        # set characters from config
        symbols, phonemes = make_symbols(**config.characters.to_dict())  # pylint: disable=redefined-outer-name
    else:
        from TTS.tts.utils.text.symbols import phonemes, symbols  # pylint: disable=import-outside-toplevel

        # use default characters and assign them to config
        config.characters = parse_symbols()
    num_chars = len(phonemes) if config.use_phonemes else len(symbols)
    # consider special `blank` character if `add_blank` is set True
    num_chars = num_chars + getattr(config, "add_blank", False)
    config.num_chars = num_chars
    # compatibility fix
    if "model_params" in config:
        config.model_params.num_chars = num_chars
    if "model_args" in config:
        config.model_args.num_chars = num_chars
    model = MyModel(config, data)
    return model


# TODO; class registery
# def import_models(models_dir, namespace):
# for file in os.listdir(models_dir):
# path = os.path.join(models_dir, file)
# if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
# model_name = file[: file.find(".py")] if file.endswith(".py") else file
# importlib.import_module(namespace + "." + model_name)
#
#
## automatically import any Python files in the models/ directory
# models_dir = os.path.dirname(__file__)
# import_models(models_dir, "TTS.tts.models")
