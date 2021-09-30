import collections
import os
import random
from typing import List, Union

import numpy as np
import torch
import torchaudio
from torch.utils.data import Dataset

from TTS.stt.datasets.tokenizer import Tokenizer
from TTS.tts.utils.data import prepare_data, prepare_tensor
from TTS.utils.audio import AudioProcessor


class FeatureExtractor:
    _FEATURE_NAMES = ["MFCC", "MEL", "LINEAR_SPEC"]

    def __init__(self, feature_name: str, ap: AudioProcessor) -> None:
        self.feature_name = feature_name
        self.ap = ap

    def __call__(self, waveform: np.ndarray):
        if self.feature_name == "MFCC":
            mfcc = self.ap.mfcc(waveform)
            return (mfcc - mfcc.mean()) / (mfcc.std() + 1e-6)
        elif self.feature_name == "MEL":
            return self.ap.melspectrogram(waveform)
        elif self.feature_name == "LINEAR_SPEC":
            return self.ap.spectrogram(waveform)
        else:
            raise ValueError("[!] Unknown feature_name: {}".format(self.feature_name))


class STTDataset(Dataset):
    def __init__(
        self,
        samples: List[List],
        ap: AudioProcessor,
        tokenizer: Tokenizer,
        batch_group_size: int = 0,
        sort_by_audio_len: bool = True,
        min_seq_len: int = 0,
        max_seq_len: int = float("inf"),
        cache_path: str = None,
        verbose: bool = False,
        feature_extractor: Union[FeatureExtractor, str] = "MFCC",
    ):

        super().__init__()
        self.samples = samples
        self.ap = ap
        self.tokenizer = tokenizer
        self.batch_group_size = batch_group_size
        self.sample_rate = ap.sample_rate
        self.min_seq_len = min_seq_len
        self.max_seq_len = max_seq_len
        self.cache_path = cache_path
        self.verbose = verbose
        self.rescue_item_idx = 1

        if isinstance(feature_extractor, str):
            self.feature_extractor = FeatureExtractor(feature_extractor, ap)
        else:
            self.feature_extractor = feature_extractor

        if cache_path:
            os.makedirs(cache_path, exist_ok=True)
        if self.verbose:
            print("\n > STT DataLoader initialization")
            print(" | > Number of instances : {}".format(len(self.samples)))

        self.sort_filter_bucket_samples(sort_by_audio_len)

    def load_wav(self, filename):
        audio = self.ap.load_wav(filename)
        return audio

    @staticmethod
    def load_np(filename):
        data = np.load(filename).astype("float32")
        return data

    def load_data(self, idx):
        sample = self.samples[idx]

        waveform = np.asarray(self.load_wav(sample["audio_file"]), dtype=np.float32)
        # waveform, sr = torchaudio.load(sample["audio_file"])
        # waveform = torchaudio.functional.resample(waveform, sr, self.ap.sample_rate)[0].numpy()

        if len(sample["text"]) > self.max_seq_len:
            # return a different sample if the phonemized
            # text is longer than the threshold
            # TODO: find a better fix
            return self.load_data(self.rescue_item_idx)

        tokens = self.tokenizer.encode(sample["text"])

        return_sample = {
            "tokens": np.array(tokens),
            "token_length": len(tokens),
            "text": sample["text"],
            "waveform": waveform,
            "audio_file": sample["audio_file"],
            "speaker_name": sample["speaker_name"],
        }
        return return_sample

    def sort_filter_bucket_samples(self, by_audio_len=False):
        r"""Sort `items` based on text length or audio length in ascending order. Filter out samples out or the length
        range.

        Args:
            by_audio_len (bool): if True, sort by audio length else by text length.
        """
        # compute the target sequence length
        if by_audio_len:
            lengths = []
            for sample in self.samples:
                lengths.append(os.path.getsize(sample["audio_file"]))
            lengths = np.array(lengths)
        else:
            lengths = np.array([len(smp[0]) for smp in self.samples])

        # sort items based on the sequence length in ascending order
        idxs = np.argsort(lengths)
        sorted_samples = []
        ignored = []
        for i, idx in enumerate(idxs):
            length = lengths[idx]
            if length < self.min_seq_len or length > self.max_seq_len:
                ignored.append(idx)
            else:
                sorted_samples.append(self.samples[idx])

        # shuffle batch groups
        # create batches with similar length items
        # the larger the `batch_group_size`, the higher the length variety in a batch.
        if self.batch_group_size > 0:
            for i in range(len(sorted_samples) // self.batch_group_size):
                offset = i * self.batch_group_size
                end_offset = offset + self.batch_group_size
                temp_items = sorted_samples[offset:end_offset]
                random.shuffle(temp_items)
                sorted_samples[offset:end_offset] = temp_items

        if len(sorted_samples) == 0:
            raise RuntimeError(" [!] No items left after filtering.")

        # update items to the new sorted items
        self.samples = sorted_samples

        # logging
        if self.verbose:
            print(" | > Max length sequence: {}".format(np.max(lengths)))
            print(" | > Min length sequence: {}".format(np.min(lengths)))
            print(" | > Avg length sequence: {}".format(np.mean(lengths)))
            print(
                " | > Num. instances discarded by max-min (max={}, min={}) seq limits: {}".format(
                    self.max_seq_len, self.min_seq_len, len(ignored)
                )
            )
            print(" | > Batch group size: {}.".format(self.batch_group_size))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.load_data(idx)

    @staticmethod
    def _sort_batch(batch, text_lengths):
        """Sort the batch by the input text lengths.

        Args:
            batch (Dict): Batch returned by `__getitem__`.
            text_lengths (List[int]): Lengths of the input character sequences.
        """
        text_lengths, ids_sorted_decreasing = torch.sort(torch.LongTensor(text_lengths), dim=0, descending=True)
        batch = [batch[idx] for idx in ids_sorted_decreasing]
        return batch, text_lengths, ids_sorted_decreasing

    def collate_fn(self, batch):
        r"""
        Perform preprocessing and create a final data batch:
        1. Sort batch instances by text-length
        2. Convert Audio signal to features.
        3. PAD sequences wrt r.
        4. Load to Torch.
        """

        # Puts each data field into a tensor with outer dimension batch size
        if isinstance(batch[0], collections.abc.Mapping):

            text_lengths = np.array([len(d["text"]) for d in batch])

            # sort items with text input length for RNN efficiency
            batch, text_lengths, _ = self._sort_batch(batch, text_lengths)

            # convert list of dicts to dict of lists
            batch = {k: [dic[k] for dic in batch] for k in batch[0]}

            # compute features
            features = [self.feature_extractor(w).astype("float32") for w in batch["waveform"]]
            feature_lengths = [m.shape[1] for m in features]

            # PAD sequences with longest instance in the batch
            tokens = prepare_data(batch["tokens"], pad_val=self.tokenizer.pad_token_id)

            # PAD features with longest instance
            features = prepare_tensor(features, 1)

            # B x D x T --> B x T x D
            features = features.transpose(0, 2, 1)

            # convert things to pytorch
            text_lengths = torch.LongTensor(text_lengths)
            tokens = torch.LongTensor(tokens)
            token_lengths = torch.LongTensor(batch["token_length"])
            features = torch.FloatTensor(features).contiguous()
            feature_lengths = torch.LongTensor(feature_lengths)

            # format waveforms wrt to computed features
            # TODO: not sure if this is good for MFCC
            wav_lengths = [w.shape[0] for w in batch["waveform"]]
            max_wav_len = max(feature_lengths) * self.ap.hop_length
            wav_lengths = torch.LongTensor(wav_lengths)
            wav_padded = torch.zeros(len(batch["waveform"]), 1, max_wav_len)
            for i, w in enumerate(batch["waveform"]):
                mel_length = feature_lengths[i]
                w = np.pad(w, (0, self.ap.hop_length), mode="edge")
                w = w[: mel_length * self.ap.hop_length]
                wav_padded[i, :, : w.shape[0]] = torch.from_numpy(w)
            wav_padded.transpose_(1, 2)
            return {
                "tokens": tokens,
                "token_lengths": token_lengths,
                "text": batch["text"],
                "text_lengths": text_lengths,
                "speaker_names": batch["speaker_name"],
                "features": features,  # (B x T x C)
                "feature_lengths": feature_lengths,  # (B)
                "audio_file": batch["audio_file"],
                "waveform": wav_padded,  # (B x T x 1)
                "raw_text": batch["text"],
            }

        raise TypeError(
            (
                "batch must contain tensors, numbers, dicts or lists;\
                         found {}".format(
                    type(batch[0])
                )
            )
        )


if __name__ == "__main__":
    from torch.utils.data.dataloader import DataLoader

    from TTS.config import BaseAudioConfig
    from TTS.stt.datasets.formatters import librispeech
    from TTS.stt.datasets.tokenizer import Tokenizer, extract_vocab

    samples = librispeech("/home/ubuntu/librispeech/LibriSpeech/train-clean-100/")
    texts = [s["text"] for s in samples]
    vocab, vocab_dict = extract_vocab(texts)
    print(vocab)
    tokenizer = Tokenizer(vocab_dict)
    ap = AudioProcessor(**BaseAudioConfig(sample_rate=16000))
    dataset = STTDataset(samples, ap=ap, tokenizer=tokenizer)

    for batch in dataset:
        print(batch["text"])
        break

    loader = DataLoader(dataset, batch_size=2, collate_fn=dataset.collate_fn)

    for batch in loader:
        print(batch)
        breakpoint()
