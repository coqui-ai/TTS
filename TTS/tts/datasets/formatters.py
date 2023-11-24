import os
import re
import xml.etree.ElementTree as ET
from glob import glob
from pathlib import Path
from typing import List

import pandas as pd
from tqdm import tqdm

########################
# DATASETS
########################


def cml_tts(root_path, meta_file, ignored_speakers=None):
    """Normalizes the CML-TTS meta data file to TTS format
    https://github.com/freds0/CML-TTS-Dataset/"""
    filepath = os.path.join(root_path, meta_file)
    # ensure there are 4 columns for every line
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
    num_cols = len(lines[0].split("|"))  # take the first row as reference
    for idx, line in enumerate(lines[1:]):
        if len(line.split("|")) != num_cols:
            print(f" > Missing column in line {idx + 1} -> {line.strip()}")
    # load metadata
    metadata = pd.read_csv(os.path.join(root_path, meta_file), sep="|")
    assert all(x in metadata.columns for x in ["wav_filename", "transcript"])
    client_id = None if "client_id" in metadata.columns else "default"
    emotion_name = None if "emotion_name" in metadata.columns else "neutral"
    items = []
    not_found_counter = 0
    for row in metadata.itertuples():
        if client_id is None and ignored_speakers is not None and row.client_id in ignored_speakers:
            continue
        audio_path = os.path.join(root_path, row.wav_filename)
        if not os.path.exists(audio_path):
            not_found_counter += 1
            continue
        items.append(
            {
                "text": row.transcript,
                "audio_file": audio_path,
                "speaker_name": client_id if client_id is not None else row.client_id,
                "emotion_name": emotion_name if emotion_name is not None else row.emotion_name,
                "root_path": root_path,
            }
        )
    if not_found_counter > 0:
        print(f" | > [!] {not_found_counter} files not found")
    return items


def coqui(root_path, meta_file, ignored_speakers=None):
    """Interal dataset formatter."""
    filepath = os.path.join(root_path, meta_file)
    # ensure there are 4 columns for every line
    with open(filepath, "r", encoding="utf8") as f:
        lines = f.readlines()
    num_cols = len(lines[0].split("|"))  # take the first row as reference
    for idx, line in enumerate(lines[1:]):
        if len(line.split("|")) != num_cols:
            print(f" > Missing column in line {idx + 1} -> {line.strip()}")
    # load metadata
    metadata = pd.read_csv(os.path.join(root_path, meta_file), sep="|")
    assert all(x in metadata.columns for x in ["audio_file", "text"])
    speaker_name = None if "speaker_name" in metadata.columns else "coqui"
    emotion_name = None if "emotion_name" in metadata.columns else "neutral"
    items = []
    not_found_counter = 0
    for row in metadata.itertuples():
        if speaker_name is None and ignored_speakers is not None and row.speaker_name in ignored_speakers:
            continue
        audio_path = os.path.join(root_path, row.audio_file)
        if not os.path.exists(audio_path):
            not_found_counter += 1
            continue
        items.append(
            {
                "text": row.text,
                "audio_file": audio_path,
                "speaker_name": speaker_name if speaker_name is not None else row.speaker_name,
                "emotion_name": emotion_name if emotion_name is not None else row.emotion_name,
                "root_path": root_path,
            }
        )
    if not_found_counter > 0:
        print(f" | > [!] {not_found_counter} files not found")
    return items


def tweb(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalize TWEB dataset.
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "tweb"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("\t")
            wav_file = os.path.join(root_path, cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def mozilla(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def mozilla_de(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, "r", encoding="ISO 8859-1") as ttf:
        for line in ttf:
            cols = line.strip().split("|")
            wav_file = cols[0].strip()
            text = cols[1].strip()
            folder_name = f"BATCH_{wav_file.split('_')[0]}_FINAL"
            wav_file = os.path.join(root_path, folder_name, wav_file)
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def mailabs(root_path, meta_files=None, ignored_speakers=None):
    """Normalizes M-AI-Labs meta data files to TTS format

    Args:
        root_path (str): root folder of the MAILAB language folder.
        meta_files (str):  list of meta files to be used in the training. If None, finds all the csv files
            recursively. Defaults to None
    """
    speaker_regex = re.compile(f"by_book{os.sep}(male|female){os.sep}(?P<speaker_name>[^{os.sep}]+){os.sep}")
    if not meta_files:
        csv_files = glob(root_path + f"{os.sep}**{os.sep}metadata.csv", recursive=True)
    else:
        csv_files = meta_files

    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for csv_file in csv_files:
        if os.path.isfile(csv_file):
            txt_file = csv_file
        else:
            txt_file = os.path.join(root_path, csv_file)

        folder = os.path.dirname(txt_file)
        # determine speaker based on folder structure...
        speaker_name_match = speaker_regex.search(txt_file)
        if speaker_name_match is None:
            continue
        speaker_name = speaker_name_match.group("speaker_name")
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_name in ignored_speakers:
                continue
        print(" | > {}".format(csv_file))
        with open(txt_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("|")
                if not meta_files:
                    wav_file = os.path.join(folder, "wavs", cols[0] + ".wav")
                else:
                    wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), "wavs", cols[0] + ".wav")
                if os.path.isfile(wav_file):
                    text = cols[1].strip()
                    items.append(
                        {"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path}
                    )
                else:
                    # M-AI-Labs have some missing samples, so just print the warning
                    print("> File %s does not exist!" % (wav_file))
    return items


def ljspeech(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
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
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def ljspeech_test(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the LJSpeech meta data file for TTS testing
    https://keithito.com/LJ-Speech-Dataset/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        speaker_id = 0
        for idx, line in enumerate(ttf):
            # 2 samples per speaker to avoid eval split issues
            if idx % 2 == 0:
                speaker_id += 1
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2]
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": f"ljspeech-{speaker_id}", "root_path": root_path}
            )
    return items


def thorsten(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the thorsten meta data file to TTS format
    https://github.com/thorstenMueller/deep-learning-german-tts/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "thorsten"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def sam_accenture(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the sam-accenture meta data file to TTS format
    https://github.com/Sam-Accenture-Non-Binary-Voice/non-binary-voice-files"""
    xml_file = os.path.join(root_path, "voice_over_recordings", meta_file)
    xml_root = ET.parse(xml_file).getroot()
    items = []
    speaker_name = "sam_accenture"
    for item in xml_root.findall("./fileid"):
        text = item.text
        wav_file = os.path.join(root_path, "vo_voice_quality_transformation", item.get("id") + ".wav")
        if not os.path.exists(wav_file):
            print(f" [!] {wav_file} in metafile does not exist. Skipping...")
            continue
        items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def ruslan(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the RUSLAN meta data file to TTS format
    https://ruslan-corpus.github.io/"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ruslan"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "RUSLAN", cols[0] + ".wav")
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def css10(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the CSS10 dataset file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "css10"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def nancy(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "nancy"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            utt_id = line.split()[1]
            text = line[line.find('"') + 1 : line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", utt_id + ".wav")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def common_voice(root_path, meta_file, ignored_speakers=None):
    """Normalize the common voice meta data file to TTS format."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_name in ignored_speakers:
                    continue
            wav_file = os.path.join(root_path, "clips", cols[1].replace(".mp3", ".wav"))
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": "MCV_" + speaker_name, "root_path": root_path}
            )
    return items


def libri_tts(root_path, meta_files=None, ignored_speakers=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if not meta_files:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    else:
        if isinstance(meta_files, str):
            meta_files = [os.path.join(root_path, meta_files)]

    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split(".")[0]
        with open(meta_file, "r", encoding="utf-8") as ttf:
            for line in ttf:
                cols = line.split("\t")
                file_name = cols[0]
                speaker_name, chapter_id, *_ = cols[0].split("_")
                _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
                wav_file = os.path.join(_root_path, file_name + ".wav")
                text = cols[2]
                # ignore speakers
                if isinstance(ignored_speakers, list):
                    if speaker_name in ignored_speakers:
                        continue
                items.append(
                    {
                        "text": text,
                        "audio_file": wav_file,
                        "speaker_name": f"LTTS_{speaker_name}",
                        "root_path": root_path,
                    }
                )
    for item in items:
        assert os.path.exists(item["audio_file"]), f" [!] wav files don't exist - {item['audio_file']}"
    return items


def custom_turkish(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "turkish-female"
    skipped_files = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0].strip() + ".wav")
            if not os.path.exists(wav_file):
                skipped_files.append(wav_file)
                continue
            text = cols[1].strip()
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    print(f" [!] {len(skipped_files)} files skipped. They don't exist...")
    return items


# ToDo: add the dataset link when the dataset is released publicly
def brspeech(root_path, meta_file, ignored_speakers=None):
    """BRSpeech 3.0 beta"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("wav_filename"):
                continue
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[2]
            speaker_id = cols[3]
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_id in ignored_speakers:
                    continue
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_id, "root_path": root_path})
    return items


def vctk(root_path, meta_files=None, wavs_path="wav48_silence_trimmed", mic="mic1", ignored_speakers=None):
    """VCTK dataset v0.92.

    URL:
        https://datashare.ed.ac.uk/bitstream/handle/10283/3443/VCTK-Corpus-0.92.zip

    This dataset has 2 recordings per speaker that are annotated with ```mic1``` and ```mic2```.
    It is believed that (😄 ) ```mic1``` files are the same as the previous version of the dataset.

    mic1:
        Audio recorded using an omni-directional microphone (DPA 4035).
        Contains very low frequency noises.
        This is the same audio released in previous versions of VCTK:
        https://doi.org/10.7488/ds/1994

    mic2:
        Audio recorded using a small diaphragm condenser microphone with
        very wide bandwidth (Sennheiser MKH 800).
        Two speakers, p280 and p315 had technical issues of the audio
        recordings using MKH 800.
    """
    file_ext = "flac"
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        # p280 has no mic2 recordings
        if speaker_id == "p280":
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_mic1.{file_ext}")
        else:
            wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + f"_{mic}.{file_ext}")
        if os.path.exists(wav_file):
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": "VCTK_" + speaker_id, "root_path": root_path}
            )
        else:
            print(f" [!] wav files don't exist - {wav_file}")
    return items


def vctk_old(root_path, meta_files=None, wavs_path="wav48", ignored_speakers=None):
    """homepages.inf.ed.ac.uk/jyamagis/release/VCTK-Corpus.tar.gz"""
    items = []
    meta_files = glob(f"{os.path.join(root_path,'txt')}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readlines()[0]
        wav_file = os.path.join(root_path, wavs_path, speaker_id, file_id + ".wav")
        items.append(
            {"text": text, "audio_file": wav_file, "speaker_name": "VCTK_old_" + speaker_id, "root_path": root_path}
        )
    return items


def synpaflex(root_path, metafiles=None, **kwargs):  # pylint: disable=unused-argument
    items = []
    speaker_name = "synpaflex"
    root_path = os.path.join(root_path, "")
    wav_files = glob(f"{root_path}**/*.wav", recursive=True)
    for wav_file in wav_files:
        if os.sep + "wav" + os.sep in wav_file:
            txt_file = wav_file.replace("wav", "txt")
        else:
            txt_file = os.path.join(
                os.path.dirname(wav_file), "txt", os.path.basename(wav_file).replace(".wav", ".txt")
            )
        if os.path.exists(txt_file) and os.path.exists(wav_file):
            with open(txt_file, "r", encoding="utf-8") as file_text:
                text = file_text.readlines()[0]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def open_bible(root_path, meta_files="train", ignore_digits_sentences=True, ignored_speakers=None):
    """ToDo: Refer the paper when available"""
    items = []
    split_dir = meta_files
    meta_files = glob(f"{os.path.join(root_path, split_dir)}/**/*.txt", recursive=True)
    for meta_file in meta_files:
        _, speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            text = file_text.readline().replace("\n", "")
        # ignore sentences that contains digits
        if ignore_digits_sentences and any(map(str.isdigit, text)):
            continue
        wav_file = os.path.join(root_path, split_dir, speaker_id, file_id + ".flac")
        items.append({"text": text, "audio_file": wav_file, "speaker_name": "OB_" + speaker_id, "root_path": root_path})
    return items


def mls(root_path, meta_files=None, ignored_speakers=None):
    """http://www.openslr.org/94/"""
    items = []
    with open(os.path.join(root_path, meta_files), "r", encoding="utf-8") as meta:
        for line in meta:
            file, text = line.split("\t")
            text = text[:-1]
            speaker, book, *_ = file.split("_")
            wav_file = os.path.join(root_path, os.path.dirname(meta_files), "audio", speaker, book, file + ".wav")
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker in ignored_speakers:
                    continue
            items.append(
                {"text": text, "audio_file": wav_file, "speaker_name": "MLS_" + speaker, "root_path": root_path}
            )
    return items


# ======================================== VOX CELEB ===========================================
def voxceleb2(root_path, meta_file=None, **kwargs):  # pylint: disable=unused-argument
    """
    :param meta_file   Used only for consistency with load_tts_samples api
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="2")


def voxceleb1(root_path, meta_file=None, **kwargs):  # pylint: disable=unused-argument
    """
    :param meta_file   Used only for consistency with load_tts_samples api
    """
    return _voxcel_x(root_path, meta_file, voxcel_idx="1")


def _voxcel_x(root_path, meta_file, voxcel_idx):
    assert voxcel_idx in ["1", "2"]
    expected_count = 148_000 if voxcel_idx == "1" else 1_000_000
    voxceleb_path = Path(root_path)
    cache_to = voxceleb_path / f"metafile_voxceleb{voxcel_idx}.csv"
    cache_to.parent.mkdir(exist_ok=True)

    # if not exists meta file, crawl recursively for 'wav' files
    if meta_file is not None:
        with open(str(meta_file), "r", encoding="utf-8") as f:
            return [x.strip().split("|") for x in f.readlines()]

    elif not cache_to.exists():
        cnt = 0
        meta_data = []
        wav_files = voxceleb_path.rglob("**/*.wav")
        for path in tqdm(
            wav_files,
            desc=f"Building VoxCeleb {voxcel_idx} Meta file ... this needs to be done only once.",
            total=expected_count,
        ):
            speaker_id = str(Path(path).parent.parent.stem)
            assert speaker_id.startswith("id")
            text = None  # VoxCel does not provide transciptions, and they are not needed for training the SE
            meta_data.append(f"{text}|{path}|voxcel{voxcel_idx}_{speaker_id}\n")
            cnt += 1
        with open(str(cache_to), "w", encoding="utf-8") as f:
            f.write("".join(meta_data))
        if cnt < expected_count:
            raise ValueError(f"Found too few instances for Voxceleb. Should be around {expected_count}, is: {cnt}")

    with open(str(cache_to), "r", encoding="utf-8") as f:
        return [x.strip().split("|") for x in f.readlines()]


def emotion(root_path, meta_file, ignored_speakers=None):
    """Generic emotion dataset"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            if line.startswith("file_path"):
                continue
            cols = line.split(",")
            wav_file = os.path.join(root_path, cols[0])
            speaker_id = cols[1]
            emotion_id = cols[2].replace("\n", "")
            # ignore speakers
            if isinstance(ignored_speakers, list):
                if speaker_id in ignored_speakers:
                    continue
            items.append(
                {"audio_file": wav_file, "speaker_name": speaker_id, "emotion_name": emotion_id, "root_path": root_path}
            )
    return items


def baker(root_path: str, meta_file: str, **kwargs) -> List[List[str]]:  # pylint: disable=unused-argument
    """Normalizes the Baker meta data file to TTS format

    Args:
        root_path (str): path to the baker dataset
        meta_file (str): name of the meta dataset containing names of wav to select and the transcript of the sentence
    Returns:
        List[List[str]]: List of (text, wav_path, speaker_name) associated with each sentences
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "baker"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            wav_name, text = line.rstrip("\n").split("|")
            wav_path = os.path.join(root_path, "clips_22", wav_name)
            items.append({"text": text, "audio_file": wav_path, "speaker_name": speaker_name, "root_path": root_path})
    return items


def kokoro(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Japanese single-speaker dataset from https://github.com/kaiidams/Kokoro-Speech-Dataset"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "kokoro"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, "wavs", cols[0] + ".wav")
            text = cols[2].replace(" ", "")
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def kss(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    """Korean single-speaker dataset from https://www.kaggle.com/datasets/bryanpark/korean-single-speaker-speech-dataset"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "kss"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[2]  # cols[1] => 6월, cols[2] => 유월
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items


def bel_tts_formatter(root_path, meta_file, **kwargs):  # pylint: disable=unused-argument
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "bel_tts"
    with open(txt_file, "r", encoding="utf-8") as ttf:
        for line in ttf:
            cols = line.split("|")
            wav_file = os.path.join(root_path, cols[0])
            text = cols[1]
            items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_name, "root_path": root_path})
    return items
