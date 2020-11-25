import os
import glob
import torch
import random
import numpy as np
from torch.utils.data import Dataset
from multiprocessing import Manager


class WaveGradDataset(Dataset):
    """
    WaveGrad Dataset searchs for all the wav files under root path
    and converts them to acoustic features on the fly and returns
    random segments of (audio, feature) couples.
    """
    def __init__(self,
                 ap,
                 items,
                 seq_len,
                 hop_len,
                 pad_short,
                 conv_pad=2,
                 is_training=True,
                 return_segments=True,
                 use_noise_augment=False,
                 use_cache=False,
                 verbose=False):

        self.ap = ap
        self.item_list = items
        self.seq_len = seq_len if return_segments else None
        self.hop_len = hop_len
        self.pad_short = pad_short
        self.conv_pad = conv_pad
        self.is_training = is_training
        self.return_segments = return_segments
        self.use_cache = use_cache
        self.use_noise_augment = use_noise_augment
        self.verbose = verbose

        if return_segments:
            assert seq_len % hop_len == 0, " [!] seq_len has to be a multiple of hop_len."
        self.feat_frame_len = seq_len // hop_len + (2 * conv_pad)

        # cache acoustic features
        if use_cache:
            self.create_feature_cache()

    def create_feature_cache(self):
        self.manager = Manager()
        self.cache = self.manager.list()
        self.cache += [None for _ in range(len(self.item_list))]

    @staticmethod
    def find_wav_files(path):
        return glob.glob(os.path.join(path, '**', '*.wav'), recursive=True)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, idx):
        item = self.load_item(idx)
        return item

    def load_test_samples(self, num_samples):
        samples = []
        return_segments = self.return_segments
        self.return_segments = False
        for idx in range(num_samples):
            mel, audio = self.load_item(idx)
            samples.append([mel, audio])
        self.return_segments = return_segments
        return samples

    def load_item(self, idx):
        """ load (audio, feat) couple """
        # compute features from wav
        wavpath = self.item_list[idx]

        if self.use_cache and self.cache[idx] is not None:
            audio = self.cache[idx]
        else:
            audio = self.ap.load_wav(wavpath)

            if self.return_segments:
                # correct audio length wrt segment length
                if audio.shape[-1] < self.seq_len + self.pad_short:
                    audio = np.pad(audio, (0, self.seq_len + self.pad_short - len(audio)), \
                            mode='constant', constant_values=0.0)
                assert audio.shape[-1] >= self.seq_len + self.pad_short, f"{audio.shape[-1]} vs {self.seq_len + self.pad_short}"

            # correct the audio length wrt hop length
            p = (audio.shape[-1] // self.hop_len + 1) * self.hop_len - audio.shape[-1]
            audio = np.pad(audio, (0, p), mode='constant', constant_values=0.0)

            if self.use_cache:
                self.cache[idx] = audio

        if self.return_segments:
            max_start = len(audio) - self.seq_len
            start = random.randint(0, max_start)
            end = start + self.seq_len
            audio = audio[start:end]

        if self.use_noise_augment and self.is_training and self.return_segments:
            audio = audio + (1 / 32768) * torch.randn_like(audio)

        mel = self.ap.melspectrogram(audio)
        mel = mel[..., :-1]  # ignore the padding

        audio = torch.from_numpy(audio).float()
        mel = torch.from_numpy(mel).float().squeeze(0)
        return (mel, audio)

    @staticmethod
    def collate_full_clips(batch):
        """This is used in tune_wavegrad.py.
        It pads sequences to the max length."""
        max_mel_length = max([b[0].shape[1] for b in batch]) if len(batch) > 1 else batch[0][0].shape[1]
        max_audio_length = max([b[1].shape[0] for b in batch]) if len(batch) > 1 else batch[0][1].shape[0]

        mels = torch.zeros([len(batch), batch[0][0].shape[0], max_mel_length])
        audios = torch.zeros([len(batch), max_audio_length])

        for idx, b in enumerate(batch):
            mel = b[0]
            audio = b[1]
            mels[idx, :, :mel.shape[1]] = mel
            audios[idx, :audio.shape[0]] = audio

        return mels, audios
