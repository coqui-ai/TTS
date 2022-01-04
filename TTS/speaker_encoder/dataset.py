import random

import numpy as np
import torch
from torch.utils.data import Dataset

from TTS.speaker_encoder.utils.generic_utils import AugmentWAV, Storage


class SpeakerEncoderDataset(Dataset):
    def __init__(
        self,
        ap,
        meta_data,
        voice_len=1.6,
        num_speakers_in_batch=64,
        storage_size=1,
        sample_from_storage_p=0.5,
        num_utter_per_speaker=10,
        skip_speakers=False,
        verbose=False,
        augmentation_config=None,
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
        self.num_speakers_in_batch = num_speakers_in_batch
        self.num_utter_per_speaker = num_utter_per_speaker
        self.skip_speakers = skip_speakers
        self.ap = ap
        self.verbose = verbose
        self.__parse_items()
        storage_max_size = storage_size * num_speakers_in_batch
        self.storage = Storage(
            maxsize=storage_max_size, storage_batchs=storage_size, num_speakers_in_batch=num_speakers_in_batch
        )
        self.sample_from_storage_p = float(sample_from_storage_p)

        speakers_aux = list(self.speakers)
        speakers_aux.sort()
        self.speakerid_to_classid = {key: i for i, key in enumerate(speakers_aux)}

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
            print(f" | > Speakers per Batch: {num_speakers_in_batch}")
            print(f" | > Storage Size: {storage_max_size} instances, each with {num_utter_per_speaker} utters")
            print(f" | > Sample_from_storage_p : {self.sample_from_storage_p}")
            print(f" | > Number of instances : {len(self.items)}")
            print(f" | > Sequence length: {self.seq_len}")
            print(f" | > Num speakers: {len(self.speakers)}")

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename, sr=self.ap.sample_rate)
        return audio

    def load_data(self, idx):
        text, wav_file, speaker_name = self.items[idx]
        wav = np.asarray(self.load_wav(wav_file), dtype=np.float32)
        mel = self.ap.melspectrogram(wav).astype("float32")
        # sample seq_len

        assert text.size > 0, self.items[idx][1]
        assert wav.size > 0, self.items[idx][1]

        sample = {
            "mel": mel,
            "item_idx": self.items[idx][1],
            "speaker_name": speaker_name,
        }
        return sample

    def __parse_items(self):
        self.speaker_to_utters = {}
        for i in self.items:
            path_ = i[1]
            speaker_ = i[2]
            if speaker_ in self.speaker_to_utters.keys():
                self.speaker_to_utters[speaker_].append(path_)
            else:
                self.speaker_to_utters[speaker_] = [
                    path_,
                ]

        if self.skip_speakers:
            self.speaker_to_utters = {
                k: v for (k, v) in self.speaker_to_utters.items() if len(v) >= self.num_utter_per_speaker
            }

        self.speakers = [k for (k, v) in self.speaker_to_utters.items()]

    def __len__(self):
        return int(1e10)

    def get_num_speakers(self):
        return len(self.speakers)

    def __sample_speaker(self, ignore_speakers=None):
        speaker = random.sample(self.speakers, 1)[0]
        # if list of speakers_id is provide make sure that it's will be ignored
        if ignore_speakers and self.speakerid_to_classid[speaker] in ignore_speakers:
            while True:
                speaker = random.sample(self.speakers, 1)[0]
                if self.speakerid_to_classid[speaker] not in ignore_speakers:
                    break

        if self.num_utter_per_speaker > len(self.speaker_to_utters[speaker]):
            utters = random.choices(self.speaker_to_utters[speaker], k=self.num_utter_per_speaker)
        else:
            utters = random.sample(self.speaker_to_utters[speaker], self.num_utter_per_speaker)
        return speaker, utters

    def __sample_speaker_utterances(self, speaker):
        """
        Sample all M utterances for the given speaker.
        """
        wavs = []
        labels = []
        for _ in range(self.num_utter_per_speaker):
            # TODO:dummy but works
            while True:
                # remove speakers that have num_utter less than 2
                if len(self.speaker_to_utters[speaker]) > 1:
                    utter = random.sample(self.speaker_to_utters[speaker], 1)[0]
                else:
                    if speaker in self.speakers:
                        self.speakers.remove(speaker)

                    speaker, _ = self.__sample_speaker()
                    continue

                wav = self.load_wav(utter)
                if wav.shape[0] - self.seq_len > 0:
                    break

                if utter in self.speaker_to_utters[speaker]:
                    self.speaker_to_utters[speaker].remove(utter)

            if self.augmentator is not None and self.data_augmentation_p:
                if random.random() < self.data_augmentation_p:
                    wav = self.augmentator.apply_one(wav)

            wavs.append(wav)
            labels.append(self.speakerid_to_classid[speaker])
        return wavs, labels

    def __getitem__(self, idx):
        speaker, _ = self.__sample_speaker()
        speaker_id = self.speakerid_to_classid[speaker]
        return speaker, speaker_id

    def __load_from_disk_and_storage(self, speaker):
        # don't sample from storage, but from HDD
        wavs_, labels_ = self.__sample_speaker_utterances(speaker)
        # put the newly loaded item into storage
        self.storage.append((wavs_, labels_))
        return wavs_, labels_

    def collate_fn(self, batch):
        # get the batch speaker_ids
        batch = np.array(batch)
        speakers_id_in_batch = set(batch[:, 1].astype(np.int32))

        labels = []
        feats = []
        speakers = set()

        for speaker, speaker_id in batch:
            speaker_id = int(speaker_id)

            # ensure that an speaker appears only once in the batch
            if speaker_id in speakers:

                # remove current speaker
                if speaker_id in speakers_id_in_batch:
                    speakers_id_in_batch.remove(speaker_id)

                speaker, _ = self.__sample_speaker(ignore_speakers=speakers_id_in_batch)
                speaker_id = self.speakerid_to_classid[speaker]
                speakers_id_in_batch.add(speaker_id)

            if random.random() < self.sample_from_storage_p and self.storage.full():
                # sample from storage (if full)
                wavs_, labels_ = self.storage.get_random_sample_fast()

                # force choose the current speaker or other not in batch
                # It's necessary for ideal training with AngleProto and GE2E losses
                if labels_[0] in speakers_id_in_batch and labels_[0] != speaker_id:
                    attempts = 0
                    while True:
                        wavs_, labels_ = self.storage.get_random_sample_fast()
                        if labels_[0] == speaker_id or labels_[0] not in speakers_id_in_batch:
                            break

                        attempts += 1
                        # Try 5 times after that load from disk
                        if attempts >= 5:
                            wavs_, labels_ = self.__load_from_disk_and_storage(speaker)
                            break
            else:
                # don't sample from storage, but from HDD
                wavs_, labels_ = self.__load_from_disk_and_storage(speaker)

            # append speaker for control
            speakers.add(labels_[0])

            # remove current speaker and append other
            if speaker_id in speakers_id_in_batch:
                speakers_id_in_batch.remove(speaker_id)

            speakers_id_in_batch.add(labels_[0])

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
                mel = self.ap.melspectrogram(wav)
                feats_.append(torch.FloatTensor(mel))

            labels.append(torch.LongTensor(labels_))
            feats.extend(feats_)

        feats = torch.stack(feats)
        labels = torch.stack(labels)

        return feats, labels
