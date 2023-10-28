import os
import random
import sys

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data
import torchaudio
from torchaudio.backend.soundfile_backend import load as torchaudio_soundfile_load
from torchaudio.backend.sox_io_backend import load as torchaudio_sox_load

torch.set_num_threads(1)


def key_samples_by_col(samples, col):
    """Returns a dictionary of samples keyed by language."""
    samples_by_col = {}
    for sample in samples:
        col_val = sample[col]
        assert isinstance(col_val, str)
        if col_val not in samples_by_col:
            samples_by_col[col_val] = []
        samples_by_col[col_val].append(sample)
    return samples_by_col


def get_prompt_slice(gt_path, max_sample_length, min_sample_length, sample_rate, is_eval=False):
    rel_clip = load_audio(gt_path, sample_rate)
    # if eval uses a middle size sample when it is possible to be more reproducible
    if is_eval:
        sample_length = int((min_sample_length + max_sample_length) / 2)
    else:
        sample_length = random.randint(min_sample_length, max_sample_length)
    gap = rel_clip.shape[-1] - sample_length
    if gap < 0:
        sample_length = rel_clip.shape[-1] // 2
    gap = rel_clip.shape[-1] - sample_length

    # if eval start always from the position 0 to be more reproducible
    if is_eval:
        rand_start = 0
    else:
        rand_start = random.randint(0, gap)

    rand_end = rand_start + sample_length
    rel_clip = rel_clip[:, rand_start:rand_end]
    rel_clip = F.pad(rel_clip, pad=(0, max_sample_length - rel_clip.shape[-1]))
    cond_idxs = [rand_start, rand_end]
    return rel_clip, rel_clip.shape[-1], cond_idxs


def load_audio(audiopath, sampling_rate):
    # better load setting following: https://github.com/faroit/python_audio_loading_benchmark
    if audiopath[-4:] == ".mp3":
        # it uses torchaudio with sox backend to load mp3
        audio, lsr = torchaudio_sox_load(audiopath)
    else:
        # it uses torchaudio soundfile backend to load all the others data type
        audio, lsr = torchaudio_soundfile_load(audiopath)

    # stereo to mono if needed
    if audio.size(0) != 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    if lsr != sampling_rate:
        audio = torchaudio.functional.resample(audio, lsr, sampling_rate)

    # Check some assumptions about audio range. This should be automatically fixed in load_wav_to_torch, but might not be in some edge cases, where we should squawk.
    # '10' is arbitrarily chosen since it seems like audio will often "overdrive" the [-1,1] bounds.
    if torch.any(audio > 10) or not torch.any(audio < 0):
        print(f"Error with {audiopath}. Max={audio.max()} min={audio.min()}")
    # clip audio invalid values
    audio.clip_(-1, 1)
    return audio


class XTTSDataset(torch.utils.data.Dataset):
    def __init__(self, config, samples, tokenizer, sample_rate, is_eval=False):
        self.config = config
        model_args = config.model_args
        self.failed_samples = set()
        self.debug_failures = model_args.debug_loading_failures
        self.max_conditioning_length = model_args.max_conditioning_length
        self.min_conditioning_length = model_args.min_conditioning_length
        self.is_eval = is_eval
        self.tokenizer = tokenizer
        self.sample_rate = sample_rate
        self.max_wav_len = model_args.max_wav_length
        self.max_text_len = model_args.max_text_length
        assert self.max_wav_len is not None and self.max_text_len is not None

        self.samples = samples
        if not is_eval:
            random.seed(config.training_seed)
            # random.shuffle(self.samples)
            random.shuffle(self.samples)
            # order by language
            self.samples = key_samples_by_col(self.samples, "language")
            print(" > Sampling by language:", self.samples.keys())
        else:
            # for evaluation load and check samples that are corrupted to ensures the reproducibility
            self.check_eval_samples()

    def check_eval_samples(self):
        print(" > Filtering invalid eval samples!!")
        new_samples = []
        for sample in self.samples:
            try:
                tseq, _, wav, _, _, _ = self.load_item(sample)
            except:
                pass
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            if (
                wav is None
                or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
                or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
            ):
                continue
            new_samples.append(sample)
        self.samples = new_samples
        print(" > Total eval samples after filtering:", len(self.samples))

    def get_text(self, text, lang):
        tokens = self.tokenizer.encode(text, lang)
        tokens = torch.IntTensor(tokens)
        assert not torch.any(tokens == 1), f"UNK token found in {text} -> {self.tokenizer.decode(tokens)}"
        # The stop token should always be sacred.
        assert not torch.any(tokens == 0), f"Stop token found in {text}"
        return tokens

    def load_item(self, sample):
        text = str(sample["text"])
        tseq = self.get_text(text, sample["language"])
        audiopath = sample["audio_file"]
        wav = load_audio(audiopath, self.sample_rate)
        if text is None or len(text.strip()) == 0:
            raise ValueError
        if wav is None or wav.shape[-1] < (0.5 * self.sample_rate):
            # Ultra short clips are also useless (and can cause problems within some models).
            raise ValueError

        # get a slice from GT to condition the model
        cond, cond_len, cond_idxs = get_prompt_slice(
            audiopath, self.max_conditioning_length, self.min_conditioning_length, self.sample_rate, self.is_eval
        )

        return tseq, audiopath, wav, cond, cond_len, cond_idxs

    def __getitem__(self, index):
        if self.is_eval:
            sample = self.samples[index]
            sample_id = str(index)
        else:
            # select a random language
            lang = random.choice(list(self.samples.keys()))
            # select random sample
            index = random.randint(0, len(self.samples[lang]) - 1)
            sample = self.samples[lang][index]
            # a unique id for each sampel to deal with fails
            sample_id = lang + "_" + str(index)

        # ignore samples that we already know that is not valid ones
        if sample_id in self.failed_samples:
            if self.debug_failures:
                print(f"Ignoring sample {sample['audio_file']} because it was already ignored before !!")
            # call get item again to get other sample
            return self[1]

        # try to load the sample, if fails added it to the failed samples list
        try:
            tseq, audiopath, wav, cond, cond_len, cond_idxs = self.load_item(sample)
        except:
            if self.debug_failures:
                print(f"error loading {sample['audio_file']} {sys.exc_info()}")
            self.failed_samples.add(sample_id)
            return self[1]

        # check if the audio and text size limits and if it out of the limits, added it failed_samples
        if (
            wav is None
            or (self.max_wav_len is not None and wav.shape[-1] > self.max_wav_len)
            or (self.max_text_len is not None and tseq.shape[0] > self.max_text_len)
        ):
            # Basically, this audio file is nonexistent or too long to be supported by the dataset.
            # It's hard to handle this situation properly. Best bet is to return the a random valid token and skew the dataset somewhat as a result.
            if self.debug_failures and wav is not None and tseq is not None:
                print(
                    f"error loading {sample['audio_file']}: ranges are out of bounds; {wav.shape[-1]}, {tseq.shape[0]}"
                )
            self.failed_samples.add(sample_id)
            return self[1]

        res = {
            # 'real_text': text,
            "text": tseq,
            "text_lengths": torch.tensor(tseq.shape[0], dtype=torch.long),
            "wav": wav,
            "wav_lengths": torch.tensor(wav.shape[-1], dtype=torch.long),
            "filenames": audiopath,
            "conditioning": cond.unsqueeze(1),
            "cond_lens": torch.tensor(cond_len, dtype=torch.long),
            "cond_idxs": torch.tensor(cond_idxs),
        }
        return res

    def __len__(self):
        if self.is_eval:
            return len(self.samples)
        return sum([len(v) for v in self.samples.values()])

    def collate_fn(self, batch):
        # convert list of dicts to dict of lists
        B = len(batch)

        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        # stack for features that already have the same shape
        batch["wav_lengths"] = torch.stack(batch["wav_lengths"])
        batch["text_lengths"] = torch.stack(batch["text_lengths"])
        batch["conditioning"] = torch.stack(batch["conditioning"])
        batch["cond_lens"] = torch.stack(batch["cond_lens"])
        batch["cond_idxs"] = torch.stack(batch["cond_idxs"])
        max_text_len = batch["text_lengths"].max()
        max_wav_len = batch["wav_lengths"].max()

        # create padding tensors
        text_padded = torch.IntTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, max_wav_len)

        # initialize tensors for zero padding
        text_padded = text_padded.zero_()
        wav_padded = wav_padded.zero_()
        for i in range(B):
            text = batch["text"][i]
            text_padded[i, : batch["text_lengths"][i]] = torch.IntTensor(text)
            wav = batch["wav"][i]
            wav_padded[i, :, : batch["wav_lengths"][i]] = torch.FloatTensor(wav)

        batch["wav"] = wav_padded
        batch["padded_text"] = text_padded
        return batch
