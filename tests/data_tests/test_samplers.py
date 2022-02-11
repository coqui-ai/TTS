import functools

import unittest

import torch

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.languages import get_language_balancer_weights
from TTS.tts.utils.speakers import get_speaker_balancer_weights

# Fixing random state to avoid random fails
torch.manual_seed(0)

dataset_config_en = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="en",
)

dataset_config_pt = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="pt-br",
)

# Adding the EN samples twice to create a language unbalanced dataset
train_samples, eval_samples = load_tts_samples(
    [dataset_config_en, dataset_config_en, dataset_config_pt], eval_split=True
)

# gerenate a speaker unbalanced dataset
for i, sample in enumerate(train_samples):
    if i < 5:
        sample[2] = "ljspeech-0"
    else:
        sample[2] = "ljspeech-1"


def is_balanced(lang_1, lang_2):
    return 0.85 < lang_1 / lang_2 < 1.2


class TestSamplers(unittest.TestCase):
    def test_language_random_sampler(self):  # pylint: disable=no-self-use
        random_sampler = torch.utils.data.RandomSampler(train_samples)
        ids = functools.reduce(lambda a, b: a + b, [list(random_sampler) for i in range(100)])
        en, pt = 0, 0
        for index in ids:
            if train_samples[index][3] == "en":
                en += 1
            else:
                pt += 1

        assert not is_balanced(en, pt), "Random sampler is supposed to be unbalanced"

    def test_language_weighted_random_sampler(self):  # pylint: disable=no-self-use
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(get_language_balancer_weights(train_samples), len(train_samples))
        ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
        en, pt = 0, 0
        for index in ids:
            if train_samples[index][3] == "en":
                en += 1
            else:
                pt += 1

        assert is_balanced(en, pt), "Language Weighted sampler is supposed to be balanced"

    def test_speaker_weighted_random_sampler(self):  # pylint: disable=no-self-use

        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(get_speaker_balancer_weights(train_samples), len(train_samples))
        ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
        spk1, spk2 = 0, 0
        for index in ids:
            if train_samples[index][2] == "ljspeech-0":
                spk1 += 1
            else:
                spk2 += 1

        assert is_balanced(spk1, spk2), "Speaker Weighted sampler is supposed to be balanced"
