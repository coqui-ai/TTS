from TTS.utils.generic_utils import find_module


def setup_model(config: "Coqpit") -> "BaseTTS":
    print(" > Using model: {}".format(config.model))
    # fetch the right model implementation.
    MyModel = find_module("TTS.enhancer.models", config.model.lower())
    model = MyModel.init_from_config(config)
    return model
