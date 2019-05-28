import os


def tweb(root_path, meta_file):
    """Normalize TWEB dataset. 
    https://www.kaggle.com/bryanpark/the-world-english-bible-speech-dataset
    """
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('\t')
            wav_file = os.path.join(root_path, cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file])
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


def mozilla_old(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            batch_no = int(cols[1].strip().split("_")[0])
            wav_folder = "batch{}".format(batch_no)
            wav_file = os.path.join(root_path, wav_folder, "wavs_no_processing", cols[1].strip())
            text = cols[0].strip()
            items.append([text, wav_file])
    return items


def mozilla(root_path, meta_file):
    """Normalizes Mozilla meta data files to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = cols[1].strip()
            text = cols[0].strip()
            wav_file = os.path.join(root_path, "wavs", wav_file)
            items.append([text, wav_file])
    return items


def mailabs(root_path, meta_files):
    """Normalizes M-AI-Labs meta data files to TTS format"""
    folders = [os.path.dirname(f.strip()) for f in meta_files]
    # meta_files = [f.strip() for f in meta_files.split(",")]
    items = []
    for idx, meta_file in enumerate(meta_files):
        print(" | > {}".format(meta_file))
        folder = folders[idx]
        txt_file = os.path.join(root_path, meta_file)
        with open(txt_file, 'r') as ttf:
            for line in ttf:
                cols = line.split('|')
                wav_file = os.path.join(root_path, folder, 'wavs',
                                        cols[0] + '.wav')
                if os.path.isfile(wav_file):
                    text = cols[1]
                    items.append([text, wav_file])
                else:
                    continue
    return items


def ljspeech(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            cols = line.split('|')
            wav_file = os.path.join(root_path, 'wavs', cols[0] + '.wav')
            text = cols[1]
            items.append([text, wav_file])
    return items


def nancy(root_path, meta_file):
    """Normalizes the Nancy meta data file to TTS format"""
    txt_file = os.path.join(root_path, meta_file)
    items = []
    with open(txt_file, 'r') as ttf:
        for line in ttf:
            id = line.split()[1]
            text = line[line.find('"') + 1:line.rfind('"') - 1]
            wav_file = os.path.join(root_path, "wavn", id + ".wav")
            items.append([text, wav_file])
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
            wav_file = os.path.join(root_path, "clips", cols[1] + ".wav")
            items.append([text, wav_file])
    return items
