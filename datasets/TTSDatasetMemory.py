import os
import random
import numpy as np
import collections
import librosa
import torch
from tqdm import tqdm
from torch.utils.data import Dataset

from utils.text import text_to_sequence
from datasets.preprocess import tts_cache
from utils.data import (prepare_data, pad_per_step, prepare_tensor,
                        prepare_stop_target)


class MyDataset(Dataset):
    # TODO: Merge to TTSDataset.py, but it is not fast as it is supposed to be
    def __init__(self,
                 root_path,
                 meta_file,
                 outputs_per_step,
                 text_cleaner,
                 ap,
                 batch_group_size=0,                 
                 min_seq_len=0,
                 **kwargs
                 ):
        self.root_path = root_path
        self.batch_group_size = batch_group_size
        self.feat_dir = os.path.join(root_path, 'loader_data')
        self.items = tts_cache(root_path, meta_file)
        self.outputs_per_step = outputs_per_step
        self.sample_rate = ap.sample_rate
        self.cleaners = text_cleaner
        self.min_seq_len = min_seq_len
        self.wavs = None
        self.mels = None
        self.linears = None
        print(" > Reading LJSpeech from - {}".format(root_path))
        print(" | > Number of instances : {}".format(len(self.items)))
        self.sort_items()
        self.fill_data()

    def fill_data(self):
        if self.wavs is None and self.mels is None:
            self.wavs = []
            self.mels = []
            self.linears = []
            self.texts = []
            for item in tqdm(self.items):
                wav_file = item[0]
                mel_file = item[1]
                linear_file = item[2]
                text = item[-1]
                wav = self.load_np(wav_file)
                mel = self.load_np(mel_file)
                linear = self.load_np(linear_file)
                self.wavs.append(wav)
                self.mels.append(mel)
                self.linears.append(linear)
                self.texts.append(np.asarray(
                    text_to_sequence(text, [self.cleaners]), dtype=np.int32))
            print(" > Data loaded to memory")

    def load_wav(self, filename):
        try:
            audio = librosa.core.load(filename, sr=self.sample_rate)
            return audio
        except RuntimeError as e:
            print(" !! Cannot read file : {}".format(filename))

    def load_np(self, filename):
        data = np.load(filename).astype('float32')
        return data

    def sort_items(self):
        r"""Sort text sequences in ascending order"""
        lengths = np.array([len(ins[-1]) for ins in self.items])

        print(" | > Max length sequence {}".format(np.max(lengths)))
        print(" | > Min length sequence {}".format(np.min(lengths)))
        print(" | > Avg length sequence {}".format(np.mean(lengths)))

        idxs = np.argsort(lengths)
        new_frames = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len:
                ignored.append(idx)
            else:
                new_frames.append(self.items[idx])
        print(" | > {} instances are ignored by min_seq_len ({})".format(
            len(ignored), self.min_seq_len))
        # shuffle batch groups
        if self.batch_group_size > 0:
            print(" | > Batch group shuffling is active.")
            for i in range(len(new_frames) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_frames = new_frames[offset : end_offset]
                random.shuffle(temp_frames)
                new_frames[offset : end_offset] = temp_frames
        self.items = new_frames

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        text = self.texts[idx]
        wav = self.wavs[idx]
        mel = self.mels[idx]
        linear = self.linears[idx]
        sample = {
            'text': text,
            'wav': wav,
            'item_idx': self.items[idx][0],
            'mel': mel,
            'linear': linear
        }
        return sample

    def collate_fn(self, batch):
        r"""
            Perform preprocessing and create a final data batch:
            1. PAD sequences with the longest sequence in the batch
            2. Convert Audio signal to Spectrograms.
            3. PAD sequences that can be divided by r.
            4. Convert Numpy to Torch tensors.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.Mapping):
            keys = list()

            wav = [d['wav'] for d in batch]
            item_idxs = [d['item_idx'] for d in batch]
            text = [d['text'] for d in batch]
            mel = [d['mel'] for d in batch]
            linear = [d['linear'] for d in batch]

            text_lenghts = np.array([len(x) for x in text])
            max_text_len = np.max(text_lenghts)
            mel_lengths = [m.shape[1] + 1 for m in mel]  # +1 for zero-frame

            # compute 'stop token' targets
            stop_targets = [
                np.array([0.] * (mel_len - 1)) for mel_len in mel_lengths
            ]

            # PAD stop targets
            stop_targets = prepare_stop_target(stop_targets,
                                               self.outputs_per_step)

            # PAD sequences with largest length of the batch
            text = prepare_data(text).astype(np.int32)
            wav = prepare_data(wav)

            # PAD features with largest length + a zero frame
            linear = prepare_tensor(linear, self.outputs_per_step)
            mel = prepare_tensor(mel, self.outputs_per_step)
            timesteps = mel.shape[2]

            # B x T x D
            linear = linear.transpose(0, 2, 1)
            mel = mel.transpose(0, 2, 1)

            # convert things to pytorch
            text_lenghts = torch.LongTensor(text_lenghts)
            text = torch.LongTensor(text)
            linear = torch.FloatTensor(linear)
            mel = torch.FloatTensor(mel)
            mel_lengths = torch.LongTensor(mel_lengths)
            stop_targets = torch.FloatTensor(stop_targets)

            return text, text_lenghts, linear, mel, mel_lengths, stop_targets, item_idxs

        raise TypeError(("batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(type(batch[0]))))
