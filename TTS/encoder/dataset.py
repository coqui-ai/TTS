import random

import numpy as np
import torch
from torch.utils.data import Dataset

from TTS.encoder.utils.generic_utils import AugmentWAV, Storage


class EncoderDataset(Dataset):
    def __init__(
        self,
        ap,
        meta_data,
        voice_len=1.6,
        num_classes_in_batch=64,
        storage_size=1,
        sample_from_storage_p=0.5,
        num_utter_per_class=10,
        skip_classes=False,
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
        self.items = meta_data
        self.sample_rate = ap.sample_rate
        self.seq_len = int(voice_len * self.sample_rate)
        self.num_classes_in_batch = num_classes_in_batch
        self.num_utter_per_class = num_utter_per_class
        self.skip_classes = skip_classes
        self.ap = ap
        self.verbose = verbose
        self.use_torch_spec = use_torch_spec
        self.__parse_items()

        storage_max_size = storage_size * num_classes_in_batch
        self.storage = Storage(
            maxsize=storage_max_size, storage_batchs=storage_size, num_classes_in_batch=num_classes_in_batch
        )
        self.sample_from_storage_p = float(sample_from_storage_p)

        classes_aux = list(self.classes)
        classes_aux.sort()
        self.classname_to_classid = {key: i for i, key in enumerate(classes_aux)}

        # Augmentation
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
            print(f" | > Storage Size: {storage_max_size} instances, each with {num_utter_per_class} utters")
            print(f" | > Sample_from_storage_p : {self.sample_from_storage_p}")
            print(f" | > Number of instances : {len(self.items)}")
            print(f" | > Sequence length: {self.seq_len}")
            print(f" | > Num Classes: {len(self.classes)}")
            print(f" | > Classes: {list(self.classes)}")


    def load_wav(self, filename):
        audio = self.ap.load_wav(filename, sr=self.ap.sample_rate)
        return audio

    def __parse_items(self):
        self.class_to_utters = {}
        for i in self.items:
            path_ = i[1]
            class_name = i[2]
            if class_name in self.class_to_utters.keys():
                self.class_to_utters[class_name].append(path_)
            else:
                self.class_to_utters[class_name] = [
                    path_,
                ]

        if self.skip_classes:
            self.class_to_utters = {
                k: v for (k, v) in self.class_to_utters.items() if len(v) >= self.num_utter_per_class
            }

        self.classes = [k for (k, v) in self.class_to_utters.items()]

    def __len__(self):
        return int(1e10)

    def get_num_classes(self):
        return len(self.classes)

    def get_map_classid_to_classname(self):
        return dict((c_id, c_n) for c_n, c_id in self.classname_to_classid.items())

    def __sample_class(self, ignore_classes=None):
        class_name = random.sample(self.classes, 1)[0]
        # if list of classes_id is provide make sure that it's will be ignored
        if ignore_classes and self.classname_to_classid[class_name] in ignore_classes:
            while True:
                class_name = random.sample(self.classes, 1)[0]
                if self.classname_to_classid[class_name] not in ignore_classes:
                    break

        if self.num_utter_per_class > len(self.class_to_utters[class_name]):
            utters = random.choices(self.class_to_utters[class_name], k=self.num_utter_per_class)
        else:
            utters = random.sample(self.class_to_utters[class_name], self.num_utter_per_class)
        return class_name, utters

    def __sample_class_utterances(self, class_name):
        """
        Sample all M utterances for the given class_name.
        """
        wavs = []
        labels = []
        for _ in range(self.num_utter_per_class):
            # TODO:dummy but works
            while True:
                # remove classes that have num_utter less than 2
                if len(self.class_to_utters[class_name]) > 1:
                    utter = random.sample(self.class_to_utters[class_name], 1)[0]
                else:
                    if class_name in self.classes:
                        self.classes.remove(class_name)

                    class_name, _ = self.__sample_class()
                    continue

                wav = self.load_wav(utter)
                if wav.shape[0] - self.seq_len > 0:
                    break

                if utter in self.class_to_utters[class_name]:
                    self.class_to_utters[class_name].remove(utter)

            if self.augmentator is not None and self.data_augmentation_p:
                if random.random() < self.data_augmentation_p:
                    wav = self.augmentator.apply_one(wav)

            wavs.append(wav)
            labels.append(self.classname_to_classid[class_name])
        return wavs, labels

    def __getitem__(self, idx):
        class_name, _ = self.__sample_class()
        class_id = self.classname_to_classid[class_name]
        return class_name, class_id

    def __load_from_disk_and_storage(self, class_name):
        # don't sample from storage, but from HDD
        wavs_, labels_ = self.__sample_class_utterances(class_name)
        # put the newly loaded item into storage
        self.storage.append((wavs_, labels_))
        return wavs_, labels_

    def collate_fn(self, batch):
        # get the batch class_ids
        batch = np.array(batch)
        classes_id_in_batch = set(batch[:, 1].astype(np.int32))

        labels = []
        feats = []
        classes = set()

        for class_name, class_id in batch:
            class_id = int(class_id)

            # ensure that an class appears only once in the batch
            if class_id in classes:

                # remove current class
                if class_id in classes_id_in_batch:
                    classes_id_in_batch.remove(class_id)

                class_name, _ = self.__sample_class(ignore_classes=classes_id_in_batch)
                class_id = self.classname_to_classid[class_name]
                classes_id_in_batch.add(class_id)

            if random.random() < self.sample_from_storage_p and self.storage.full():
                # sample from storage (if full)
                wavs_, labels_ = self.storage.get_random_sample_fast()

                # force choose the current class or other not in batch
                # It's necessary for ideal training with AngleProto and GE2E losses
                if labels_[0] in classes_id_in_batch and labels_[0] != class_id:
                    attempts = 0
                    while True:
                        wavs_, labels_ = self.storage.get_random_sample_fast()
                        if labels_[0] == class_id or labels_[0] not in classes_id_in_batch:
                            break

                        attempts += 1
                        # Try 5 times after that load from disk
                        if attempts >= 5:
                            wavs_, labels_ = self.__load_from_disk_and_storage(class_name)
                            break
            else:
                # don't sample from storage, but from HDD
                wavs_, labels_ = self.__load_from_disk_and_storage(class_name)

            # append class for control
            classes.add(labels_[0])

            # remove current class and append other
            if class_id in classes_id_in_batch:
                classes_id_in_batch.remove(class_id)

            classes_id_in_batch.add(labels_[0])

            # get a random subset of each of the wavs and extract mel spectrograms.
            feats_ = []
            for wav in wavs_:
                offset = random.randint(0, wav.shape[0] - self.seq_len)
                wav = wav[offset : offset + self.seq_len]
                # add random gaussian noise
                if self.gaussian_augmentation_config and self.gaussian_augmentation_config["p"]:
                    if random.random() < self.gaussian_augmentation_config["p"]:
                        wav += np.random.normal(
                            self.gaussian_augmentation_config["min_amplitude"],
                            self.gaussian_augmentation_config["max_amplitude"],
                            size=len(wav),
                        )

                if not self.use_torch_spec:
                    mel = self.ap.melspectrogram(wav)
                    feats_.append(torch.FloatTensor(mel))
                else:
                    feats_.append(torch.FloatTensor(wav))

            labels.append(torch.LongTensor(labels_))
            feats.extend(feats_)

        feats = torch.stack(feats)
        labels = torch.stack(labels)

        return feats, labels
