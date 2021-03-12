import io
import json
import os
import zipfile
from pathlib import Path
from shutil import copyfile

import gdown
import requests
from TTS.utils.generic_utils import get_user_data_dir
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
    def __init__(self, models_file=None, output_prefix=None):
        super().__init__()
        if output_prefix is None:
            self.output_prefix = get_user_data_dir('tts')
        else:
            self.output_prefix = os.path.join(output_prefix, 'tts')
        self.url_prefix = "https://drive.google.com/uc?id="
        self.models_dict = None
        if models_file is not None:
            self.read_models_file(models_file)
        else:
            # try the default location
            path = Path(__file__).parent / "../.models.json"
            self.read_models_file(path)

    def read_models_file(self, file_path):
        """Read .models.json as a dict

        Args:
            file_path (str): path to .models.json.
        """
        with open(file_path, "r", encoding="utf-8") as json_file:
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
        models_name_list = []
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    for model in self.models_dict[model_type][lang][dataset]:
                        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
                        output_path = os.path.join(self.output_prefix, model_full_name)
                        if os.path.exists(output_path):
                            print(f" >: {model_type}/{lang}/{dataset}/{model} [already downloaded]")
                        else:
                            print(f" >: {model_type}/{lang}/{dataset}/{model}")
                        models_name_list.append(f'{model_type}/{lang}/{dataset}/{model}')
        return models_name_list

    def download_model(self, model_name):
        """Download model files given the full model name.
        Model name is in the format
            'type/language/dataset/model'
            e.g. 'tts_model/en/ljspeech/tacotron'

        Every model must have the following files
            - *.pth.tar : pytorch model checkpoint file.
            - config.json : model config file.
            - scale_stats.npy (if exist): scale values for preprocessing.

        Args:
            model_name (str): model name as explained above.

        TODO: support multi-speaker models
        """
        # fetch model info from the dict
        model_type, lang, dataset, model = model_name.split("/")
        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
        model_item = self.models_dict[model_type][lang][dataset][model]
        # set the model specific output path
        output_path = os.path.join(self.output_prefix, model_full_name)
        output_model_path = os.path.join(output_path, "model_file.pth.tar")
        output_config_path = os.path.join(output_path, "config.json")
        if os.path.exists(output_path):
            print(f" > {model_name} is already downloaded.")
        else:
            os.makedirs(output_path, exist_ok=True)
            print(f" > Downloading model to {output_path}")
            output_stats_path = os.path.join(output_path, 'scale_stats.npy')
            # download files to the output path
            if self._check_dict_key(model_item, 'github_rls_url'):
                # download from github release
                # TODO: pass output_path
                self._download_zip_file(model_item['github_rls_url'], output_path)
            else:
                # download from gdrive
                self._download_gdrive_file(model_item['model_file'], output_model_path)
                self._download_gdrive_file(model_item['config_file'], output_config_path)
                if self._check_dict_key(model_item, 'stats_file'):
                    self._download_gdrive_file(model_item['stats_file'], output_stats_path)

            # set the scale_path.npy file path in the model config.json
            if self._check_dict_key(model_item, 'stats_file') or os.path.exists(output_stats_path):
                # set scale stats path in config.json
                config_path = output_config_path
                config = load_config(config_path)
                config["audio"]['stats_path'] = output_stats_path
                with open(config_path, "w") as jf:
                    json.dump(config, jf)
        return output_model_path, output_config_path, model_item

    def _download_gdrive_file(self, gdrive_idx, output):
        """Download files from GDrive using their file ids"""
        gdown.download(f"{self.url_prefix}{gdrive_idx}", output=output, quiet=False)

    @staticmethod
    def _download_zip_file(file_url, output):
        """Download the github releases"""
        r = requests.get(file_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(output)
        for file_path in z.namelist()[1:]:
            src_path = os.path.join(output, file_path)
            dst_path = os.path.join(output, os.path.basename(file_path))
            copyfile(src_path, dst_path)

    @staticmethod
    def _check_dict_key(my_dict, key):
        if key in my_dict.keys() and my_dict[key] is not None:
            if not isinstance(key, str):
                return True
            if isinstance(key, str) and len(my_dict[key]) > 0:
                return True
        return False
