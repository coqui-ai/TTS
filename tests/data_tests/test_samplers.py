import functools

import torch

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.languages import get_language_weighted_sampler

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
    if train_samples[index][3] == "en":
        en += 1
    else:
        pt += 1

assert not is_balanced(en, pt), "Random sampler is supposed to be unbalanced"

weighted_sampler = get_language_weighted_sampler(train_samples)
ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
en, pt = 0, 0
for index in ids:
    if train_samples[index][3] == "en":
        en += 1
    else:
        pt += 1

assert is_balanced(en, pt), "Weighted sampler is supposed to be balanced"
