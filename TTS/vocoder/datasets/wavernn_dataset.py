import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset


class WaveRNNDataset(Dataset):
    """
    WaveRNN Dataset searchs for all the wav files under root path
    and converts them to acoustic features on the fly.
    """

    def __init__(
        self,
        ap,
        items,
        seq_len,
        hop_len,
        pad,
        mode,
        is_training=True,
        return_segments=True,
        use_cache=False,
        verbose=False,
    ):

        self.ap = ap
        self.item_list = items
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.pad = pad
        self.mode = mode
        self.is_training = is_training
        self.return_segments = return_segments
        self.use_cache = use_cache
        self.verbose = verbose

        # wav_files = [f"{self.path}wavs/{file}.wav" for file in self.metadata]
        # with Pool(4) as pool:
        #    self.wav_cache = pool.map(self.ap.load_wav, wav_files)

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_item(self, index):
        wavpath, feat_path = self.item_list[index]
        m = np.load(feat_path.replace("/quant/", "/mel/"))
        # x = self.wav_cache[index]
        if 5 > m.shape[-1]:
            print(" [!] Instance is too short! : {}".format(wavpath))
            self.item_list[index] = self.item_list[index + 1]
            feat_path = self.item_list[index]
            m = np.load(feat_path.replace("/quant/", "/mel/"))
        if self.mode in ["gauss", "mold"]:
            x = self.ap.load_wav(wavpath)
        elif isinstance(self.mode, int):
            x = np.load(feat_path.replace("/mel/", "/quant/"))
        else:
            raise RuntimeError("Unknown dataset mode - ", self.mode)
        return m, x

    def collate(self, batch):
        mel_win = self.seq_len // self.hop_len + 2 * self.pad
        max_offsets = [x[0].shape[-1] - (mel_win + 2 * self.pad) for x in batch]
        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + self.pad) * self.hop_len for offset in mel_offsets]

        mels = [
            x[0][:, mel_offsets[i] : mel_offsets[i] + mel_win]
            for i, x in enumerate(batch)
        ]

        coarse = [
            x[1][sig_offsets[i] : sig_offsets[i] + self.seq_len + 1]
            for i, x in enumerate(batch)
        ]

        mels = np.stack(mels).astype(np.float32)
        if self.mode in ["gauss", "mold"]:
            coarse = np.stack(coarse).astype(np.float32)
            coarse = torch.FloatTensor(coarse)
            x_input = coarse[:, : self.seq_len]
        elif isinstance(self.mode, int):
            coarse = np.stack(coarse).astype(np.int64)
            coarse = torch.LongTensor(coarse)
            x_input = (
                2 * coarse[:, : self.seq_len].float() / (2 ** self.mode - 1.0) - 1.0
            )
        y_coarse = coarse[:, 1:]
        mels = torch.FloatTensor(mels)
        return x_input, mels, y_coarse
