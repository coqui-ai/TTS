import numpy as np
import torch
from torch.utils.data import Dataset

from TTS.utils.audio.numpy_transforms import mulaw_encode, quantize


class WaveRNNDataset(Dataset):
    """
    WaveRNN Dataset searchs for all the wav files under root path
    and converts them to acoustic features on the fly.
    """

    def __init__(
        self, ap, items, seq_len, hop_len, pad, mode, mulaw, is_training=True, verbose=False, return_segments=True
    ):
        super().__init__()
        self.ap = ap
        self.compute_feat = not isinstance(items[0], (tuple, list))
        self.item_list = items
        self.seq_len = seq_len
        self.hop_len = hop_len
        self.mel_len = seq_len // hop_len
        self.pad = pad
        self.mode = mode
        self.mulaw = mulaw
        self.is_training = is_training
        self.verbose = verbose
        self.return_segments = return_segments

        assert self.seq_len % self.hop_len == 0

    def __len__(self):
        return len(self.item_list)

    def __getitem__(self, index):
        item = self.load_item(index)
        return item

    def load_test_samples(self, num_samples):
        samples = []
        return_segments = self.return_segments
        self.return_segments = False
        for idx in range(num_samples):
            mel, audio, _ = self.load_item(idx)
            samples.append([mel, audio])
        self.return_segments = return_segments
        return samples

    def load_item(self, index):
        """
        load (audio, feat) couple if feature_path is set
        else compute it on the fly
        """
        if self.compute_feat:
            wavpath = self.item_list[index]
            audio = self.ap.load_wav(wavpath)
            if self.return_segments:
                min_audio_len = 2 * self.seq_len + (2 * self.pad * self.hop_len)
            else:
                min_audio_len = audio.shape[0] + (2 * self.pad * self.hop_len)
            if audio.shape[0] < min_audio_len:
                print(" [!] Instance is too short! : {}".format(wavpath))
                audio = np.pad(audio, [0, min_audio_len - audio.shape[0] + self.hop_len])
            mel = self.ap.melspectrogram(audio)

            if self.mode in ["gauss", "mold"]:
                x_input = audio
            elif isinstance(self.mode, int):
                x_input = (
                    mulaw_encode(wav=audio, mulaw_qc=self.mode)
                    if self.mulaw
                    else quantize(x=audio, quantize_bits=self.mode)
                )
            else:
                raise RuntimeError("Unknown dataset mode - ", self.mode)

        else:
            wavpath, feat_path = self.item_list[index]
            mel = np.load(feat_path.replace("/quant/", "/mel/"))

            if mel.shape[-1] < self.mel_len + 2 * self.pad:
                print(" [!] Instance is too short! : {}".format(wavpath))
                self.item_list[index] = self.item_list[index + 1]
                feat_path = self.item_list[index]
                mel = np.load(feat_path.replace("/quant/", "/mel/"))
            if self.mode in ["gauss", "mold"]:
                x_input = self.ap.load_wav(wavpath)
            elif isinstance(self.mode, int):
                x_input = np.load(feat_path.replace("/mel/", "/quant/"))
            else:
                raise RuntimeError("Unknown dataset mode - ", self.mode)

        return mel, x_input, wavpath

    def collate(self, batch):
        mel_win = self.seq_len // self.hop_len + 2 * self.pad
        max_offsets = [x[0].shape[-1] - (mel_win + 2 * self.pad) for x in batch]

        mel_offsets = [np.random.randint(0, offset) for offset in max_offsets]
        sig_offsets = [(offset + self.pad) * self.hop_len for offset in mel_offsets]

        mels = [x[0][:, mel_offsets[i] : mel_offsets[i] + mel_win] for i, x in enumerate(batch)]

        coarse = [x[1][sig_offsets[i] : sig_offsets[i] + self.seq_len + 1] for i, x in enumerate(batch)]

        mels = np.stack(mels).astype(np.float32)
        if self.mode in ["gauss", "mold"]:
            coarse = np.stack(coarse).astype(np.float32)
            coarse = torch.FloatTensor(coarse)
            x_input = coarse[:, : self.seq_len]
        elif isinstance(self.mode, int):
            coarse = np.stack(coarse).astype(np.int64)
            coarse = torch.LongTensor(coarse)
            x_input = 2 * coarse[:, : self.seq_len].float() / (2**self.mode - 1.0) - 1.0
        y_coarse = coarse[:, 1:]
        mels = torch.FloatTensor(mels)
        return x_input, mels, y_coarse
