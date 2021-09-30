import os
import re
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import List

from tqdm import tqdm

########################
# DATASETS
########################


def ljspeech(root_path, meta_file):
    """Normalizes the LJSpeech meta data file to TTS format
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name})
    return items


def librispeech(root_path, meta_files=None):
    """http://www.openslr.org/resources/12/"""
    _delimiter = " "
    _audio_ext = ".flac"
    items = []
    if meta_files is None:
        meta_files = glob(f"{root_path}/**/*trans.txt", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split(_delimiter, 1)
                file_name = cols[0]
                speaker_name, chapter_id, *_ = cols[0].split("-")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + _audio_ext)
                text = cols[1]
                items.append({"text": text, "audio_file": wav_file, "speaker_name": "LibriSpeech_" + speaker_name})
    for item in items:
        assert os.path.exists(item["audio_file"]), f" [!] wav files don't exist - {item['audio_file']}"
    return items


if __name__ == "__main__":
    items_ = librispeech(root_path="/home/ubuntu/librispeech/pforLibriSpeech/train-clean-100/", meta_files=None)
