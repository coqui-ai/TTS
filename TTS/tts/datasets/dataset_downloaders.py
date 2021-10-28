import os
from os.path import expanduser

import requests
import tarfile
import logging
from tqdm import tqdm

from TTS.utils.generic_utils import get_user_data_dir


class DatasetDownloaders:
    def __init__(
            self,
            dataset_name: str,
            output_path: str = None,
            libri_tts_subset: str = 'all',
            voxceleb_version: str = 'both',
            mailabs_language: str = 'all'
    ):
        self.name = dataset_name
        self.libri_tts_subset = libri_tts_subset
        self.voxceleb_version = voxceleb_version
        self.mailabs_language = mailabs_language
        if output_path is None:
            self.output_path = get_user_data_dir('tts/datasets')
        else:
            self.output_path = os.path.join(output_path, 'tts/datasets')

        self.dataset_dict = {
            'thorsten-de': ("https://www.openslr.org/resources/95/thorsten-de_v02.tgz",
                            "thorsten-de_v02.tgz", 'Thorsten-De'),
            'ljspeech': ("https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2",
                         "LJSpeech-1.1.tar.bz2", "LJSpeech-1.1"),
            'common-voice': (),
            'tweb': ('bryanpark/the-world-english-bible-speech-dataset', 'tweb.zip', 'Tweb'),
            'libri-tts-clean-100': ("http://www.openslr.org/resources/60/train-clean-100.tar.gz",
                                    'train-clean-100.tar.tz', 'LibriTTS-100-hours'),
            'libri-tts-clean-360': ("http://www.openslr.org/resources/60/train-clean-360.tar.gz",
                                    'train-clean-360.tar.tz', 'LibriTTS-360-hours'),
            'libri-tts-other-500': ("http://www.openslr.org/resources/60/train-other-500.tar.gz",
                                    'train-other-500.tar.tz', 'LibriTTS-500-hours'),
            'libri-tts-dev-clean': (
                "http://www.openslr.org/resources/60/dev-clean.tar.gz", 'dev-clean.tar.tz', 'LibriTTS-dev-clean'),
            'libri-tts-dev-other': (
                "http://www.openslr.org/resources/60/dev-other.tar.gz", 'dev-other.tar.tz', 'LibriTTS-dev-other'),
            'libri-tts-test-clean': (
                "http://www.openslr.org/resources/60/test-clean.tar.gz", 'test-clean.tar.tz', 'LibriTTS-test-clean'),
            'libri-tts-test-other': (
                "http://www.openslr.org/resources/60/test-other.tar.gz", 'test-other.tar.tz', 'LibriTTS-test-other'),
            'mailabs-english': ("https://data.solak.de/data/Training/stt_tts/en_US.tgz",
                                'en_US.tgz', 'MaiLabs-English'),
            'mailabs-german': ("https://data.solak.de/data/Training/stt_tts/de_DE.tgz",
                               'de_DE.tgz', 'MaiLabs-German'),
            'mailabs-french': ("https://data.solak.de/data/Training/stt_tts/fr_FR.tgz",
                               'fr_FR.tgz', 'MaiLabs-French'),
            'mailabs-italian': ("https://data.solak.de/data/Training/stt_tts/it_IT.tgz",
                                'it_IT.tgz', 'MaiLabs-Italian'),
            'mailabs-spanish': ("https://data.solak.de/data/Training/stt_tts/es_ES.tgz",
                                'es_ES.tgz', 'MaiLabs-Spanish'),
            'vctk-kaggle': ('mfekadu/english-multispeaker-corpus-for-voice-cloning', 'vctk.zip', 'Vctk'),
            'vctk': ("datashare.is.ed.ac.uk/download/DS_10283_3443.zip", 'vctk.zip', 'Vctk'),
        }

    def download_dataset(self):
        if self.name == 'ljspeech':
            dataset_path = self._download_ljspeech()
        elif self.name == 'libri-tts':
            dataset_path = self._download_libri_tts()
        elif self.name == 'thorsten-de':
            dataset_path = self._download_thorsten_german()
        elif self.name == 'mailabs':
            dataset_path = self._download_mailabs()
        elif self.name == "vctk":
            dataset_path = self._download_vctk()
        elif self.name == 'tweb':
            dataset_path = self._download_tweb()
        return dataset_path

    def list_datasets(self):
        data_list_dict = {
            'ljspeech': ('ljspeech',
                         '24 hours of professional audio of a female reading audio books.This dataset is 2.76 gigs in size.'),

        }
        print(data_list_dict)

    def _download_ljspeech(self):
        url, tar_file, data_name = self.dataset_dict['ljspeech']
        data_path = os.path.join(self.output_path, data_name)
        self._download_tarred_data(url, tar_file, data_name)
        return data_path

    def _download_thorsten_german(self):
        url, tar_file, data_name = self.dataset_dict['thorsten-de']
        data_path = os.path.join(self.output_path, data_name)
        self._download_tarred_data(url, tar_file, data_name)
        return data_path

    def _download_libri_tts(self):
        if self.libri_tts_subset == 'all':
            subset_names = ['libri-tts-clean-100', 'libri-tts-clean-360', 'libri-tts-other-500', 'libri-tts-dev-clean',
                            'libri-tts-dev-other', 'libri-tts-test-clean', 'libri-tts-test-other']
            for i in subset_names:
                url, tar_file, subset_name = self.dataset_dict[i]
                self._download_tarred_data(url, tar_file, subset_name)
            print("finished downloading all subsets")
        elif self.libri_tts_subset == 'clean':
            subset_names = ['libri-tts-clean-100', 'libri-tts-clean-360', 'libri-tts-dev-clean', 'libri-tts-test-clean']
            for i in subset_names:
                url, tar_file, subset_name = self.dataset_dict[i]
                self._download_tarred_data(url, tar_file, subset_name)
            print('finished downloading the clean subsets')
        elif self.libri_tts_subset == "noisy":
            subset_names = ['libri-tts-other-500', 'libri-tts-dev-other', 'libri-tts-test-other']
            for i in subset_names:
                url, tar_file, subset_name = self.dataset_dict[i]
                self._download_tarred_data(url, tar_file, subset_name)
            print('finished downloading the noisy subsets')
        dataset_path = os.path.join(self.output_path, "LibriTTS")
        print(f'your dataset was downloaded to {os.path.join(self.output_path, "LibriTTS")}')
        return dataset_path

    def _download_vctk(self, remove_silences=True, use_kaggle=True):
        if use_kaggle:
            url, tar_file, dataset_name = self.dataset_dict['vctk-kaggle']
            data_dir = self._download_kaggle_dataset(url, dataset_name)
            dataset_path = os.path.join(data_dir, )
            return dataset_path
        else:
            pass

    def _download_tweb(self):
        url, tar_file, dataset_name = self.dataset_dict['tweb']
        data_dir = self._download_kaggle_dataset(url, dataset_name)
        return data_dir

    def _download_mailabs(self):
        if self.mailabs_language == 'all':
            language_subsets = ['mailabs-english', 'mailabs-german',
                                'mailabs-french', 'mailabs-italian', 'mailabs-spanish']
            for subset in language_subsets:
                url, tar_file, data_name = self.dataset_dict[subset]
                self._download_tarred_data(url, tar_file, data_name, custom_dir_name="MaiLabs")
        elif self.mailabs_language == 'english':
            url, tar_file, data_name = self.dataset_dict['mailabs-english']
            self._download_tarred_data(url, tar_file, data_name, custom_dir_name="MaiLabs")
        elif self.mailabs_language == 'french':
            url, tar_file, data_name = self.dataset_dict['mailabs-french']
            self._download_tarred_data(url, tar_file, data_name)
        elif self.mailabs_language == 'german':
            url, tar_file, data_name = self.dataset_dict['mailabs-german']
            self._download_tarred_data(url, tar_file, data_name)
        elif self.mailabs_language == 'italian':
            url, tar_file, data_name = self.dataset_dict['mailabs-italian']
            self._download_tarred_data(url, tar_file, data_name)
        elif self.mailabs_language == 'spanish':
            url, tar_file, data_name = self.dataset_dict['mailabs-spanish']
            self._download_tarred_data(url, tar_file, data_name)
        print(f'your dataset was downloaded to {os.path.join(self.output_path, "MaiLabs")}')
        return data_name

    def _download_voxceleb(self, version="both"):
        pass

    def _download_common_voice(self, version):
        pass

    def _download_tarred_data(self, url, tar_file, data_name, custom_dir_name=None):
        if custom_dir_name is not None:
            dataset_dir = os.path.join(self.output_path, custom_dir_name)
        else:
            dataset_dir = self.output_path
        with open(os.path.join(self.output_path, tar_file), "wb") as data:
            raw_bytes = requests.get(url, stream=True)
            total_bytes = int(raw_bytes.headers['Content-Length'])
            progress_bar = tqdm(total=total_bytes, unit="MiM", unit_scale=True)
            print(f"\ndownloading {data_name} data")
            for chunk in raw_bytes.iter_content(chunk_size=1024):
                data.write(chunk)
                progress_bar.update(len(chunk))
        data.close()
        progress_bar.close()
        print("\nextracting data")
        tar_path = tarfile.open(os.path.join(self.output_path, tar_file))
        tar_path.extractall(dataset_dir)
        tar_path.close()
        os.remove(os.path.join(self.output_path, tar_file))

    def _download_kaggle_dataset(self, dataset_path, dataset_name):
        data_path = os.path.join(self.output_path, dataset_name)
        try:
            import kaggle
            kaggle.api.authenticate()
            print(f"""\nThe {dataset_name} dataset is being download and untarred via kaggle api. This may take a minute depending on the dataset size""")
            kaggle.api.dataset_download_files(dataset_path, path=data_path, unzip=True)
            print(f'dataset is downloaded and stored in {data_path}')
            return data_path
        except OSError:
            logging.warning(f"""in order to download kaggle datasets, you need to have a kaggle api token stored in your 
{os.path.join(expanduser('~'), '.kaggle/kaggle.json')} 
If you don't have a kaggle account you can easily make one for free and generate a token in the account settings tab.""")
