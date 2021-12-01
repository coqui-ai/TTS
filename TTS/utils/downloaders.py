import os
from typing import Optional

from TTS.utils.download import download_kaggle_dataset, download_url, extract_archive


def download_ljspeech(path: str):
    """Download and extract LJSpeech dataset

    Args:
        path (str): path to the directory where the dataset will be stored.
    """
    os.makedirs(path, exist_ok=True)
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    download_url(url, path)
    basename = os.path.basename(url)
    archive = os.path.join(path, basename)
    print(" > Extracting archive file...")
    extract_archive(archive)


def download_vctk(path: str, use_kaggle: Optional[bool] = False):
    """Download and extract VCTK dataset.

    Args:
        path (str): path to the directory where the dataset will be stored.

        use_kaggle (bool, optional): Downloads vctk dataset from kaggle. Is generally faster. Defaults to False.
    """
    if use_kaggle:
        download_kaggle_dataset("mfekadu/english-multispeaker-corpus-for-voice-cloning", "VCTK", path)
    else:
        os.makedirs(path, exist_ok=True)
        url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
        download_url(url, path)
        basename = os.path.basename(url)
        archive = os.path.join(path, basename)
        print(" > Extracting archive file...")
        extract_archive(archive)


def download_tweb(path: str):
    """Download and extract Tweb dataset

    Args:
        path (str): Path to the directory where the dataset will be stored.
    """
    download_kaggle_dataset("bryanpark/the-world-english-bible-speech-dataset", "TWEB", path)


def download_libri_tts(path: str, subset: Optional[str] = "all"):
    """Download and extract libri tts dataset.

    Args:
        path (str): Path to the directory where the dataset will be stored.

        subset (str, optional): Name of the subset to download. If you only want to download a certain
        portion specify it here. Defaults to 'all'.
    """

    subset_dict = {
        "libri-tts-clean-100": "http://www.openslr.org/resources/60/train-clean-100.tar.gz",
        "libri-tts-clean-360": "http://www.openslr.org/resources/60/train-clean-360.tar.gz",
        "libri-tts-other-500": "http://www.openslr.org/resources/60/train-other-500.tar.gz",
        "libri-tts-dev-clean": "http://www.openslr.org/resources/60/dev-clean.tar.gz",
        "libri-tts-dev-other": "http://www.openslr.org/resources/60/dev-other.tar.gz",
        "libri-tts-test-clean": "http://www.openslr.org/resources/60/test-clean.tar.gz",
        "libri-tts-test-other": "http://www.openslr.org/resources/60/test-other.tar.gz",
    }

    os.makedirs(path, exist_ok=True)
    if subset == "all":
        for sub, val in subset_dict.items():
            print(f" > Downloading {sub}...")
            download_url(val, path)
            basename = os.path.basename(val)
            archive = os.path.join(path, basename)
            print(" > Extracting archive file...")
            extract_archive(archive)
        print(" > All subsets downloaded")
    else:
        url = subset_dict[subset]
        download_url(url, path)
        basename = os.path.basename(url)
        archive = os.path.join(path, basename)
        print(" > Extracting archive file...")
        extract_archive(archive)


def download_thorsten_de(path: str):
    """Download and extract Thorsten german male voice dataset.

    Args:
        path (str): Path to the directory where the dataset will be stored.
    """
    os.makedirs(path, exist_ok=True)
    url = "https://www.openslr.org/resources/95/thorsten-de_v02.tgz"
    download_url(url, path)
    basename = os.path.basename(url)
    archive = os.path.join(path, basename)
    print(" > Extracting archive file...")
    extract_archive(archive)


def download_mailabs(path: str, language: str = "english"):
    """Download and extract Mailabs dataset.

    Args:
        path (str): Path to the directory where the dataset will be stored.

        language (str): Language subset to download. Defaults to english.
    """
    language_dict = {
        "english": "https://data.solak.de/data/Training/stt_tts/en_US.tgz",
        "german": "https://data.solak.de/data/Training/stt_tts/de_DE.tgz",
        "french": "https://data.solak.de/data/Training/stt_tts/fr_FR.tgz",
        "italian": "https://data.solak.de/data/Training/stt_tts/it_IT.tgz",
        "spanish": "https://data.solak.de/data/Training/stt_tts/es_ES.tgz",
    }
    os.makedirs(path, exist_ok=True)
    url = language_dict[language]
    download_url(url, path)
    basename = os.path.basename(url)
    archive = os.path.join(path, basename)
    print(" > Extracting archive file...")
    extract_archive(archive)
