import pandas as pd
import os
import numpy as np
import collections
from torch.utils.data import Dataset

import train_config as c
from Tacotron.text import text_to_sequence
from Tacotron.utils.audio import *
from Tacotron.utils.data import prepare_data, pad_data, pad_per_step


class LJSpeechDataset(Dataset):

    def __init__(self, csv_file, root_dir, outputs_per_step):
        self.frames = pd.read_csv(csv_file, sep='|', header=None)
        self.root_dir = root_dir
        self.outputs_per_step = outputs_per_step
        print(" > Reading LJSpeech from - {}".format(root_dir))
        print(" | > Number of instances : {}".format(len(self.frames)))

    def load_wav(self, filename):
        try:
            audio = librosa.load(filename, sr=c.sample_rate)
            return audio
        except RuntimeError as e:
            print(" !! Cannot read file : {}".format(filename))

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        wav_name = os.path.join(self.root_dir,
                                self.frames.ix[idx, 0]) + '.wav'
        text = self.frames.ix[idx, 1]
        text = np.asarray(text_to_sequence(text, [c.cleaners]), dtype=np.int32)
        wav = np.asarray(self.load_wav(wav_name)[0], dtype=np.float32)
        sample = {'text': text, 'wav': wav}
        return sample

    def collate_fn(self, batch):

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            keys = list()

            text = [d['text'] for d in batch]
            wav = [d['wav'] for d in batch]

            # PAD sequences with largest length of the batch
            text = prepare_data(text).astype(np.int32)
            wav = prepare_data(wav)

            magnitude = np.array([spectrogram(w) for w in wav])
            mel = np.array([melspectrogram(w) for w in wav])
            timesteps = mel.shape[-1]

            # PAD with zeros that can be divided by outputs per step
            if timesteps % self.outputs_per_step != 0:
                magnitude = pad_per_step(magnitude, self.outputs_per_step)
                mel = pad_per_step(mel, self.outputs_per_step)

            return text, magnitude, mel

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}"
                         .format(type(batch[0]))))
