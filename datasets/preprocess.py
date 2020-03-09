import os
from glob import glob
import re
import sys
from TTS.utils.generic_utils import split_dataset


def load_meta_data(datasets):
    meta_data_train_all = []
    meta_data_eval_all = []
    for dataset in datasets:
        name = dataset['name']
        root_path = dataset['path']
        meta_file_train = dataset['meta_file_train']
        meta_file_val = dataset['meta_file_val']
        preprocessor = get_preprocessor_by_name(name)

        meta_data_train = preprocessor(root_path, meta_file_train)
        if meta_file_val is None:
            meta_data_eval, meta_data_train = split_dataset(meta_data_train)
        else:
            meta_data_eval = preprocessor(root_path, meta_file_val)
        meta_data_train_all += meta_data_train
        meta_data_eval_all += meta_data_eval
    return meta_data_train_all, meta_data_eval_all


def get_preprocessor_by_name(name):
    """Returns the respective preprocessing function."""
    thismodule = sys.modules[__name__]
    return getattr(thismodule, name.lower())


def tweb(root_path, meta_file):
    """Normalize TWEB dataset.
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "tweb"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('\t')
            wav_file = os.path.join(root_path, cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items


# def kusal(root_path, meta_file):
#     txt_file = os.path.join(root_path, meta_file)
#     texts = []
#     wavs = []
#     with open(txt_file, "r", encoding="utf8") as f:
#         frames = [
#             line.split('\t') for line in f
#             if line.split('\t')[0] in self.wav_files_dict.keys()
#         ]
#     # TODO: code the rest
#     return  {'text': texts, 'wavs': wavs}


def mozilla(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append([text, wav_file, speaker_name])
    return items


def mozilla_de(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "mozilla"
    with open(txt_file, 'r', encoding="ISO 8859-1") as ttf:
        for line in ttf:
            cols = line.strip().split('|')
            wav_file = cols[0].strip()
            text = cols[1].strip()
            folder_name = f"BATCH_{wav_file.split('_')[0]}_FINAL"
            wav_file = os.path.join(root_path, folder_name, wav_file)
            items.append([text, wav_file, speaker_name])
    return items


def mailabs(root_path, meta_files=None):
    """Normalizes M-AI-Labs meta data files to TTS format"""
    speaker_regex = re.compile("by_book/(male|female)/(?P<speaker_name>[^/]+)/")
    if meta_files is None:
        csv_files = glob(root_path+"/**/metadata.csv", recursive=True)
    else:
        csv_files = meta_files
    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for csv_file in csv_files:
        txt_file = os.path.join(root_path, csv_file)
        folder = os.path.dirname(txt_file)
        # determine speaker based on folder structure...
        speaker_name_match = speaker_regex.search(txt_file)
        if speaker_name_match is None:
            continue
        speaker_name = speaker_name_match.group("speaker_name")
        print(" | > {}".format(csv_file))
        with open(txt_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                if meta_files is None:
                    wav_file = os.path.join(folder, 'wavs', cols[0] + '.wav')
                else:
                    wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), 'wavs', cols[0] + '.wav')
                if os.path.isfile(wav_file):
                    text = cols[1].strip()
                    items.append([text, wav_file, speaker_name])
                else:
                    raise RuntimeError("> File %s is not exist!"%(wav_file))
    return items


def ljspeech(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "ljspeech"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file, speaker_name])
    return items


def nancy(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    speaker_name = "nancy"
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            utt_id = line.split()[1]
            text = line[line.find('"') + 1:line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", utt_id + ".wav")
            items.append([text, wav_file, speaker_name])
    return items


def common_voice(root_path, meta_file):
    """Normalize the common voice meta data file to TTS format."""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            if line.startswith("client_id"):
                continue
            cols = line.split("\t")
            text = cols[2]
            speaker_name = cols[0]
            wav_file = os.path.join(root_path, "clips", cols[1] + ".wav")
            items.append([text, wav_file, speaker_name])
    return items


def libri_tts(root_path, meta_files=None):
    """https://ai.google/tools/datasets/libri-tts/"""
    items = []
    if meta_files is None:
        meta_files = glob(f"{root_path}/**/*trans.tsv", recursive=True)
    for meta_file in meta_files:
        _meta_file = os.path.basename(meta_file).split('.')[0]
        speaker_name = _meta_file.split('_')[0]
        chapter_id = _meta_file.split('_')[1]
        _root_path = os.path.join(root_path, f"{speaker_name}/{chapter_id}")
        with open(meta_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('\t')
                wav_file = os.path.join(_root_path, cols[0] + '.wav')
                text = cols[1]
                items.append([text, wav_file, speaker_name])
    for item in items:
        assert os.path.exists(item[1]), f" [!] wav file is not exist - {item[1]}"
    return items
