import importlib
import os
from inspect import isclass

# import all files under configs/
configs_dir = os.path.dirname(__file__)
for file in os.listdir(configs_dir):
    path = os.path.join(configs_dir, file)
    if not file.startswith("_") and not file.startswith(".") and (file.endswith(".py") or os.path.isdir(path)):
        config_name = file[: file.find(".py")] if file.endswith(".py") else file
        module = importlib.import_module("TTS.vocoder.configs." + config_name)
        for attribute_name in dir(module):
            attribute = getattr(module, attribute_name)

            if isclass(attribute):
                # Add the class to this package's variables
                globals()[attribute_name] = attribute
