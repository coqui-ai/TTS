import random

import torch
from torch.utils.data import Dataset
from librosa.core import load, resample
import librosa
from TTS.encoder.utils.generic_utils import AugmentWAV
from torch.nn.utils.rnn import pad_sequence
import random


class EnhancerDataset(Dataset):
    def __init__(
        self,
        config,
        ap,
        meta_data,
        verbose=False,
        augmentation_config=None,
        use_torch_spec=None,
    ):
        """
        Args:
            ap (TTS.tts.utils.AudioProcessor): audio processor object.
            meta_data (list): list of dataset instances.
            seq_len (int): voice segment length in seconds.
            verbose (bool): print diagnostic information.
        """
        super().__init__()
        self.config = config
        self.items = meta_data
        self.sample_rate = ap.sample_rate
        self.ap = ap
        self.verbose = verbose
        self.use_torch_spec = use_torch_spec
        self.input_sr = self.config.input_sr
        self.target_sr = self.config.target_sr
        self.segment_train = self.config.segment_train
        self.segment_len = self.config.segment_len

        # Data Augmentation
        self.augmentator = None
        self.gaussian_augmentation_config = None
        if augmentation_config:
            self.data_augmentation_p = augmentation_config["p"]
            if self.data_augmentation_p and ("additive" in augmentation_config or "rir" in augmentation_config):
                self.augmentator = AugmentWAV(ap, augmentation_config)

        if self.verbose:
            print("\n > DataLoader initialization")
            print(f" | > Number of instances : {len(self.items)}")
            print(f" | > Input sample rate : {self.input_sr}")
            print(f" | > Target sample rate : {self.target_sr}")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]

    def segment_wav(self, wav):
        """
        Extract a random segment of segment_len from waveform.
        Args:
            wav (np.array): waveform.
        Returns:
            np.array: random segment.
        """
        if self.segment_train:
            segment_len = int(self.segment_len * self.sample_rate)
            if segment_len > len(wav):
                return wav
            segment_start = random.randint(0, len(wav) - segment_len)
            segment_end = segment_start + segment_len
            segment = wav[segment_start:segment_end]
            return segment
        else:
            return wav

    def load_audio(self, wav_path):
        wav, sr = load(wav_path, sr=None, mono=True)
        assert sr == self.target_sr, f"Sample rate mismatch: {sr} vs {self.target_sr}"
        if self.ap.trim_silence:
            wav = self.trim_silence(wav)
        return wav

    def trim_silence(self, wav):
        margin = int(self.ap.sample_rate * 0.01)
        wav = wav[margin:-margin]
        return librosa.effects.trim(wav, top_db=self.ap.trim_db, frame_length=self.ap.win_length, hop_length=self.ap.hop_length)[0]

    def collate_fn(self, batch):
        input = []
        input_lens = []
        target = []
        target_lens = []
        for item in batch:
            audio_path = item["audio_file"]

            # load wav file
            target_wav = self.load_audio(audio_path)
            # segment wav file
            target_wav = self.segment_wav(target_wav)
            # make sure that the length of the wav is a multiple of 3
            if len(target_wav) %3 != 0:
                target_wav = target_wav[:-(len(target_wav) % 3)]
            input_wav = resample(target_wav, orig_sr=self.target_sr, target_sr=self.input_sr, res_type="zero_order_hold")

            if self.augmentator is not None and self.data_augmentation_p:
                if random.random() < self.data_augmentation_p:
                    input_wav = self.augmentator.apply_one(input_wav)
             
            input_lens.append(len(input_wav))
            target_lens.append(len(target_wav))
            input.append(torch.tensor(input_wav, dtype=torch.float32))
            target.append(torch.tensor(target_wav, dtype=torch.float32))

        return {
            "input_wav": pad_sequence(input, batch_first=True),
            "target_wav": pad_sequence(target, batch_first=True),
            "input_lens": torch.tensor(input_lens, dtype=torch.int32),
            "target_lens": torch.tensor(target_lens, dtype=torch.int32),
        }