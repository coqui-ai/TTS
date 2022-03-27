import random

import torch
from torch.utils.data import Dataset
from librosa.core import load, resample
from TTS.encoder.utils.generic_utils import AugmentWAV
from torch.nn.utils.rnn import pad_sequence


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

    def load_audio(self, wav_path):
        wav, sr = load(wav_path, sr=None, mono=True)
        assert sr == self.target_sr, f"Sample rate mismatch: {sr} vs {self.target_sr}"
        return wav

    def collate_fn(self, batch):
        input = []
        input_lens = []
        target = []
        target_lens = []
        for item in batch:
            audio_path = item["audio_file"]

            # load wav file
            target_wav = self.load_audio(audio_path)
            input_wav = resample(target_wav, self.target_sr, self.input_sr, res_type="linear")

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