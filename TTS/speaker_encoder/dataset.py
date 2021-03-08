import queue
import random

import numpy as np
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ap, meta_data, voice_len=1.6, num_speakers_in_batch=64,
                 storage_size=1, sample_from_storage_p=0.5, additive_noise=0,
                 num_utter_per_speaker=10, skip_speakers=False, verbose=False):
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
        self.voice_len = voice_len
        self.seq_len = int(voice_len * self.sample_rate)
        self.num_speakers_in_batch = num_speakers_in_batch
        self.num_utter_per_speaker = num_utter_per_speaker
        self.skip_speakers = skip_speakers
        self.ap = ap
        self.verbose = verbose
        self.__parse_items()
        self.storage = queue.Queue(maxsize=storage_size*num_speakers_in_batch)
        self.sample_from_storage_p = float(sample_from_storage_p)
        self.additive_noise = float(additive_noise)
        if self.verbose:
            print("\n > DataLoader initialization")
            print(f" | > Speakers per Batch: {num_speakers_in_batch}")
            print(f" | > Storage Size: {self.storage.maxsize} speakers, each with {num_utter_per_speaker} utters")
            print(f" | > Sample_from_storage_p : {self.sample_from_storage_p}")
            print(f" | > Noise added : {self.additive_noise}")
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
                self.speaker_to_utters[speaker_] = [path_, ]

        if self.skip_speakers:
            self.speaker_to_utters = {k: v for (k, v) in self.speaker_to_utters.items() if
                                      len(v) >= self.num_utter_per_speaker}

        self.speakers = [k for (k, v) in self.speaker_to_utters.items()]

    # def __parse_items(self):
    #     """
    #     Find unique speaker ids and create a dict mapping utterances from speaker id
    #     """
    #     speakers = list({item[-1] for item in self.items})
    #     self.speaker_to_utters = {}
    #     self.speakers = []
    #     for speaker in speakers:
    #         speaker_utters = [item[1] for item in self.items if item[2] == speaker]
    #         if len(speaker_utters) < self.num_utter_per_speaker and self.skip_speakers:
    #             print(
    #                 f" [!] Skipped speaker {speaker}. Not enough utterances {self.num_utter_per_speaker} vs {len(speaker_utters)}."
    #             )
    #         else:
    #             self.speakers.append(speaker)
    #             self.speaker_to_utters[speaker] = speaker_utters

    def __len__(self):
        return int(1e10)

    def __sample_speaker(self):
        speaker = random.sample(self.speakers, 1)[0]
        if self.num_utter_per_speaker > len(self.speaker_to_utters[speaker]):
            utters = random.choices(
                self.speaker_to_utters[speaker], k=self.num_utter_per_speaker
            )
        else:
            utters = random.sample(
                self.speaker_to_utters[speaker], self.num_utter_per_speaker
            )
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
                if len(self.speaker_to_utters[speaker]) > 0:
                    utter = random.sample(self.speaker_to_utters[speaker], 1)[0]
                else:
                    self.speakers.remove(speaker)
                    speaker, _ = self.__sample_speaker()
                    continue
                wav = self.load_wav(utter)
                if wav.shape[0] - self.seq_len > 0:
                    break
                self.speaker_to_utters[speaker].remove(utter)

            wavs.append(wav)
            labels.append(speaker)
        return wavs, labels

    def __getitem__(self, idx):
        speaker, _ = self.__sample_speaker()
        return speaker

    def collate_fn(self, batch):
        labels = []
        feats = []
        for speaker in batch:
            if random.random() < self.sample_from_storage_p and self.storage.full():
                # sample from storage (if full), ignoring the speaker
                wavs_, labels_ = random.choice(self.storage.queue)
            else:
                # don't sample from storage, but from HDD
                wavs_, labels_ = self.__sample_speaker_utterances(speaker)
                # if storage is full, remove an item
                if self.storage.full():
                    _ = self.storage.get_nowait()
                # put the newly loaded item into storage
                self.storage.put_nowait((wavs_, labels_))

            # add random gaussian noise
            if self.additive_noise > 0:
                noises_ = [np.random.normal(0, self.additive_noise, size=len(w)) for w in wavs_]
                wavs_ = [wavs_[i] + noises_[i] for i in range(len(wavs_))]

            # get a random subset of each of the wavs and convert to MFCC.
            offsets_ = [random.randint(0, wav.shape[0] - self.seq_len) for wav in wavs_]
            mels_ = [self.ap.melspectrogram(wavs_[i][offsets_[i]: offsets_[i] + self.seq_len]) for i in range(len(wavs_))]
            feats_ = [torch.FloatTensor(mel) for mel in mels_]

            labels.append(labels_)
            feats.extend(feats_)
        feats = torch.stack(feats)
        return feats.transpose(1, 2), labels
