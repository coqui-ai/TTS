import glob
import os
from multiprocessing import Pool

import librosa
import soundfile as sf
from tqdm import tqdm

from TTS.stt.utils.download import download_url, extract_archive


def _resample_file(func_args):
    filename, output_sr = func_args
    y, sr = librosa.load(filename, sr=output_sr)
    sf.write(filename, y, sr)


def download_ljspeech(path: str, split_name: str = None, n_jobs: int = 1):
    """Download and extract LJSpeech dataset and resample it to 16khz."""

    SAMPLE_RATE = 16000
    os.makedirs(path, exist_ok=True)

    # download and extract
    url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
    download_url(url, path)
    basename = os.path.basename(url)
    archive = os.path.join(path, basename)
    print(" > Extracting archive file...")
    extract_archive(archive)

    # resample wav files to SAMPLE_RAT0E
    print(" > Resampling the audio files...")
    print(os.path.join(path, "LJSpeech-1.1", "**/*.wav"))
    audio_files = glob.glob(os.path.join(path, "LJSpeech-1.1", "**/*.wav"), recursive=True)
    print(f"> Found {len(audio_files)} files...")
    audio_files = list(zip(audio_files, len(audio_files) * [SAMPLE_RATE]))
    with Pool(processes=n_jobs) as p:
        with tqdm(total=len(audio_files)) as pbar:
            for i, _ in enumerate(p.imap_unordered(_resample_file, audio_files)):
                pbar.update()


def download_librispeech(path: str, split_name: str):
    """Download and extract LibriSpeech dataset splits."""

    if split_name not in [
        "dev-clean",
        "dev-other",
        "test-clean",
        "test-other",
        "train-clean-100",
        "train-clean-360",
        "train-other-500",
    ]:
        raise ValueError("[!] `split_name` is not valid")

    os.makedirs(path, exist_ok=True)

    ext_archive = ".tar.gz"
    base_url = "http://www.openslr.org/resources/12/"
    url = os.path.join(base_url, split_name + ext_archive)
    download_url(url, path)
    basename = os.path.basename(url)
    archive = os.path.join(path, basename)
    extract_archive(archive)


if __name__ == "__main__":
    # download_librispeech("/home/ubuntu/librispeech/", "train-clean-100")
    download_ljspeech("/home/ubuntu/ljspeech/", n_jobs=8)
