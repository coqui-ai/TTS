import json
import gdown
from pathlib import Path
import os

from TTS.utils.io import load_config


class ModelManager(object):
    """Manage TTS models defined in .models.json.
    It provides an interface to list and download
    models defines in '.model.json'

    Models are downloaded under '.TTS' folder in the user's
    home path.

    Args:
        models_file (str): path to .model.json
    """
    def __init__(self, models_file):
        super().__init__()
        self.output_prefix = os.path.join(str(Path.home()), '.tts')
        self.url_prefix = "https://drive.google.com/uc?id="
        self.models_dict = None
        self.read_models_file(models_file)

    def read_models_file(self, file_path):
        """Read .models.json as a dict

        Args:
            file_path (str): path to .models.json.
        """
        with open(file_path) as json_file:
            self.models_dict = json.load(json_file)

    def list_langs(self):
        print(" Name format: type/language")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                print(f" >: {model_type}/{lang} ")

    def list_datasets(self):
        print(" Name format: type/language/dataset")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    print(f" >: {model_type}/{lang}/{dataset}")

    def list_models(self):
        print(" Name format: type/language/dataset/model")
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                   for model in self.models_dict[model_type][lang][dataset]:
                        print(f" >: {model_type}/{lang}/{dataset}/{model} ")

    def download_model(self, model_name):
        """Download model files given the full model name.
        Model name is in the format
            'type/language/dataset/model'
            e.g. 'tts_model/en/ljspeech/tacotron'

        Args:
            model_name (str): model name as explained above.

        TODO: support multi-speaker models
        """
        # fetch model info from the dict
        type, lang, dataset, model = model_name.split("/")
        model_full_name = f"{type}--{lang}--{dataset}--{model}"
        model_item = self.models_dict[type][lang][dataset][model]
        # set the model specific output path
        output_path = os.path.join(self.output_prefix, model_full_name)
        output_model_path = os.path.join(output_path, "model_file.pth.tar")
        output_config_path = os.path.join(output_path, "config.json")
        if os.path.exists(output_path):
            print(f" > {model_name} is already downloaded.")
        else:
            os.makedirs(output_path, exist_ok=True)
            print(f" > Downloading model to {output_path}")
            output_stats_path = None
            # download files to the output path
            self._download_file(model_item['model_file'], output_model_path)
            self._download_file(model_item['config_file'], output_config_path)
            if model_item['stats_file'] is not None and len(model_item['stats_file']) > 1:
                output_stats_path = os.path.join(output_path, 'scale_stats.npy')
                self._download_file(model_item['stats_file'], output_stats_path)
                # set scale stats path in config.json
                config_path = output_config_path
                config = load_config(config_path)
                config["audio"]['stats_path'] = output_stats_path
                with open(config_path, "w") as jf:
                    json.dump(config, jf)
        return output_model_path, output_config_path

    def _download_file(self, id, output):
        gdown.download(f"{self.url_prefix}{id}", output=output)






