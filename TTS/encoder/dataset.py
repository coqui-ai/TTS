import random

import torch
from torch.utils.data import Dataset

from TTS.encoder.utils.generic_utils import AugmentWAV


class EncoderDataset(Dataset):
    def __init__(
        self,
        config,
        ap,
        meta_data,
        voice_len=1.6,
        num_classes_in_batch=64,
        num_utter_per_class=10,
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
        self.seq_len = int(voice_len * self.sample_rate)
        self.num_utter_per_class = num_utter_per_class
        self.ap = ap
        self.verbose = verbose
        self.use_torch_spec = use_torch_spec
        self.classes, self.items = self.__parse_items()

        self.classname_to_classid = {key: i for i, key in enumerate(self.classes)}

        # Data Augmentation
        self.augmentator = None
        self.gaussian_augmentation_config = None
        if augmentation_config:
            self.data_augmentation_p = augmentation_config["p"]
            if self.data_augmentation_p and ("additive" in augmentation_config or "rir" in augmentation_config):
                self.augmentator = AugmentWAV(ap, augmentation_config)

            if "gaussian" in augmentation_config.keys():
                self.gaussian_augmentation_config = augmentation_config["gaussian"]

        if self.verbose:
            print("\n > DataLoader initialization")
            print(f" | > Classes per Batch: {num_classes_in_batch}")
            print(f" | > Number of instances : {len(self.items)}")
            print(f" | > Sequence length: {self.seq_len}")
            print(f" | > Num Classes: {len(self.classes)}")
            print(f" | > Classes: {self.classes}")

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename, sr=self.ap.sample_rate)
        return audio

    def __parse_items(self):
        class_to_utters = {}
        for item in self.items:
            path_ = item["audio_file"]
            class_name = item[self.config.class_name_key]
            if class_name in class_to_utters.keys():
                class_to_utters[class_name].append(path_)
            else:
                class_to_utters[class_name] = [
                    path_,
                ]

        # skip classes with number of samples >= self.num_utter_per_class
        class_to_utters = {k: v for (k, v) in class_to_utters.items() if len(v) >= self.num_utter_per_class}

        classes = list(class_to_utters.keys())
        classes.sort()

        new_items = []
        for item in self.items:
            path_ = item["audio_file"]
            class_name = item["emotion_name"] if self.config.model == "emotion_encoder" else item["speaker_name"]
            # ignore filtered classes
            if class_name not in classes:
                continue
            # ignore small audios
            if self.load_wav(path_).shape[0] - self.seq_len <= 0:
                continue

            new_items.append({"wav_file_path": path_, "class_name": class_name})

        return classes, new_items

    def __len__(self):
        return len(self.items)

    def get_num_classes(self):
        return len(self.classes)

    def get_class_list(self):
        return self.classes

    def set_classes(self, classes):
        self.classes = classes
        self.classname_to_classid = {key: i for i, key in enumerate(self.classes)}

    def get_map_classid_to_classname(self):
        return dict((c_id, c_n) for c_n, c_id in self.classname_to_classid.items())

    def __getitem__(self, idx):
        return self.items[idx]

    def collate_fn(self, batch):
        # get the batch class_ids
        labels = []
        feats = []
        for item in batch:
            utter_path = item["wav_file_path"]
            class_name = item["class_name"]

            # get classid
            class_id = self.classname_to_classid[class_name]
            # load wav file
            wav = self.load_wav(utter_path)
            offset = random.randint(0, wav.shape[0] - self.seq_len)
            wav = wav[offset : offset + self.seq_len]

            if self.augmentator is not None and self.data_augmentation_p:
                if random.random() < self.data_augmentation_p:
                    wav = self.augmentator.apply_one(wav)

            if not self.use_torch_spec:
                mel = self.ap.melspectrogram(wav)
                feats.append(torch.FloatTensor(mel))
            else:
                feats.append(torch.FloatTensor(wav))

            labels.append(class_id)

        feats = torch.stack(feats)
        labels = torch.LongTensor(labels)

        return feats, labels
