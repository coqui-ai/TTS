import os

from TTS.utils.download import download_url, extract_archive


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


def download_vctk(path: str):
    """Download and extract VCTK dataset

    Args:
        path (str): path to the directory where the dataset will be stored.
    """
    os.makedirs(path, exist_ok=True)
    url = "https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip"
    download_url(url, path)
    basename = os.path.basename(url)
    archive = os.path.join(path, basename)
    print(" > Extracting archive file...")
    extract_archive(archive)
