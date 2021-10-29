import io
import json
import os
import zipfile
from pathlib import Path
from shutil import copyfile, rmtree

import requests

from TTS.config import load_config
from TTS.utils.generic_utils import get_user_data_dir


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
            self.output_prefix = get_user_data_dir("tts")
        else:
            self.output_prefix = os.path.join(output_prefix, "tts")
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
        model_count = 1
        for model_type in self.models_dict:
            for lang in self.models_dict[model_type]:
                for dataset in self.models_dict[model_type][lang]:
                    for model in self.models_dict[model_type][lang][dataset]:
                        model_full_name = f"{model_type}--{lang}--{dataset}--{model}"
                        output_path = os.path.join(self.output_prefix, model_full_name)
                        if os.path.exists(output_path):
                            print(f" {model_count}: {model_type}/{lang}/{dataset}/{model} [already downloaded]")
                        else:
                            print(f" {model_count}: {model_type}/{lang}/{dataset}/{model}")
                        models_name_list.append(f"{model_type}/{lang}/{dataset}/{model}")
                        model_count += 1
        return models_name_list

    def download_model(self, model_name):
        """Download model files given the full model name.
        Model name is in the format
            'type/language/dataset/model'
            e.g. 'tts_model/en/ljspeech/tacotron'

        Every model must have the following files:
            - *.pth.tar : pytorch model checkpoint file.
            - config.json : model config file.
            - scale_stats.npy (if exist): scale values for preprocessing.

        Args:
            model_name (str): model name as explained above.
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
            # download from github release
            self._download_zip_file(model_item["github_rls_url"], output_path)
        # update paths in the config.json
        self._update_paths(output_path, output_config_path)
        return output_model_path, output_config_path, model_item

    def _update_paths(self, output_path: str, config_path: str) -> None:
        """Update paths for certain files in config.json after download.

        Args:
            output_path (str): local path the model is downloaded to.
            config_path (str): local config.json path.
        """
        output_stats_path = os.path.join(output_path, "scale_stats.npy")
        output_d_vector_file_path = os.path.join(output_path, "speakers.json")
        output_speaker_ids_file_path = os.path.join(output_path, "speaker_ids.json")

        # update the scale_path.npy file path in the model config.json
        self._update_path("audio.stats_path", output_stats_path, config_path)

        # update the speakers.json file path in the model config.json to the current path
        self._update_path("d_vector_file", output_d_vector_file_path, config_path)
        self._update_path("model_args.d_vector_file", output_d_vector_file_path, config_path)

        # update the speaker_ids.json file path in the model config.json to the current path
        self._update_path("speakers_file", output_speaker_ids_file_path, config_path)
        self._update_path("model_args.speakers_file", output_speaker_ids_file_path, config_path)

    @staticmethod
    def _update_path(field_name, new_path, config_path):
        """Update the path in the model config.json for the current environment after download"""
        if os.path.exists(new_path):
            config = load_config(config_path)
            field_names = field_name.split(".")
            if len(field_names) > 1:
                # field name points to a sub-level field
                sub_conf = config
                for fd in field_names[:-1]:
                    if fd in sub_conf:
                        sub_conf = sub_conf[fd]
                    else:
                        return
                sub_conf[field_names[-1]] = new_path
            else:
                # field name points to a top-level field
                config[field_name] = new_path
            config.save_json(config_path)

    @staticmethod
    def _download_zip_file(file_url, output_folder):
        """Download the github releases"""
        # download the file
        r = requests.get(file_url)
        # extract the file
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            z.extractall(output_folder)
        # move the files to the outer path
        for file_path in z.namelist()[1:]:
            src_path = os.path.join(output_folder, file_path)
            dst_path = os.path.join(output_folder, os.path.basename(file_path))
            copyfile(src_path, dst_path)
        # remove the extracted folder
        rmtree(os.path.join(output_folder, z.namelist()[0]))

    @staticmethod
    def _check_dict_key(my_dict, key):
        if key in my_dict.keys() and my_dict[key] is not None:
            if not isinstance(key, str):
                return True
            if isinstance(key, str) and len(my_dict[key]) > 0:
                return True
        return False
