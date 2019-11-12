import numpy as np
import torch
import random
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, ap, meta_data, voice_len=1.6, num_speakers_in_batch=64,
                 num_utter_per_speaker=10, skip_speakers=False, verbose=False):
        """
        Args:
            ap (TTS.utils.AudioProcessor): audio processor object.
            meta_data (list): list of dataset instances.
            seq_len (int): voice segment length in seconds.
            verbose (bool): print diagnostic information.
        """
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
        if self.verbose:
            print("\n > DataLoader initialization")
            print(f" | > Number of instances : {len(self.items)}")
            print(f" | > Sequence length: {self.seq_len}")
            print(f" | > Num speakers: {len(self.speakers)}")

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
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
        """
        Find unique speaker ids and create a dict mapping utterances from speaker id
        """
        speakers = list({item[-1] for item in self.items})
        self.speaker_to_utters = {}
        self.speakers = []
        for speaker in speakers:
            speaker_utters = [item[1] for item in self.items if item[2] == speaker]
            if len(speaker_utters) < self.num_utter_per_speaker and self.skip_speakers:
                print(
                    f" [!] Skipped speaker {speaker}. Not enough utterances {self.num_utter_per_speaker} vs {len(speaker_utters)}."
                )
            else:
                self.speakers.append(speaker)
                self.speaker_to_utters[speaker] = speaker_utters

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
        feats = []
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

            offset = random.randint(0, wav.shape[0] - self.seq_len)
            mel = self.ap.melspectrogram(wav[offset : offset + self.seq_len])
            feats.append(torch.FloatTensor(mel))
            labels.append(speaker)
        return feats, labels

    def __getitem__(self, idx):
        speaker, _ = self.__sample_speaker()
        return speaker

    def collate_fn(self, batch):
        labels = []
        feats = []
        for speaker in batch:
            feats_, labels_ = self.__sample_speaker_utterances(speaker)
            labels.append(labels_)
            feats.extend(feats_)
        feats = torch.stack(feats)
        return feats.transpose(1, 2), labels
