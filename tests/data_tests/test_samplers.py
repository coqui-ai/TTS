import functools

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

# Adding the EN samples twice to create an unbalanced dataset
train_samples, eval_samples = load_tts_samples(
    [dataset_config_en, dataset_config_en, dataset_config_pt], eval_split=True
)


def is_balanced(lang_1, lang_2):
    return 0.85 < lang_1 / lang_2 < 1.2


random_sampler = torch.utils.data.RandomSampler(train_samples)
ids = functools.reduce(lambda a, b: a + b, [list(random_sampler) for i in range(100)])
en, pt = 0, 0
for index in ids:
    if train_samples[index]["language"] == "en":
        en += 1
    else:
        pt += 1

assert not is_balanced(en, pt), "Random sampler is supposed to be unbalanced"


weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(get_language_balancer_weights(train_samples), len(train_samples))
ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
en, pt = 0, 0
for index in ids:
    if train_samples[index]["language"] == "en":
        en += 1
    else:
        pt += 1

assert is_balanced(en, pt), "Language Weighted sampler is supposed to be balanced"

# test speaker weighted sampler

# gerenate a speaker unbalanced dataset
for i in range(0, len(train_samples)):
    if i < 5:
        train_samples[i][2] = "ljspeech-0"
    else:
        train_samples[i][2] = "ljspeech-1"

weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(get_speaker_balancer_weights(train_samples), len(train_samples))
ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
spk1, spk2 = 0, 0
for index in ids:
    if train_samples[index][2] == "ljspeech-0":
        spk1 += 1
    else:
        spk2 += 1

assert is_balanced(spk1, spk2), "Speaker Weighted sampler is supposed to be balanced"
