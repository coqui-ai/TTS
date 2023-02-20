import functools
import random
import unittest

import torch

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.utils.data import get_length_balancer_weights
from TTS.tts.utils.languages import get_language_balancer_weights
from TTS.tts.utils.speakers import get_speaker_balancer_weights
from TTS.utils.samplers import BucketBatchSampler, PerfectBatchSampler

# Fixing random state to avoid random fails
torch.manual_seed(0)

dataset_config_en = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="en",
)

dataset_config_pt = BaseDatasetConfig(
    formatter="ljspeech",
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
        sample["speaker_name"] = "ljspeech-0"
    else:
        sample["speaker_name"] = "ljspeech-1"


def is_balanced(lang_1, lang_2):
    return 0.85 < lang_1 / lang_2 < 1.2


class TestSamplers(unittest.TestCase):
    def test_language_random_sampler(self):  # pylint: disable=no-self-use
        random_sampler = torch.utils.data.RandomSampler(train_samples)
        ids = functools.reduce(lambda a, b: a + b, [list(random_sampler) for i in range(100)])
        en, pt = 0, 0
        for index in ids:
            if train_samples[index]["language"] == "en":
                en += 1
            else:
                pt += 1

        assert not is_balanced(en, pt), "Random sampler is supposed to be unbalanced"

    def test_language_weighted_random_sampler(self):  # pylint: disable=no-self-use
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            get_language_balancer_weights(train_samples), len(train_samples)
        )
        ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
        en, pt = 0, 0
        for index in ids:
            if train_samples[index]["language"] == "en":
                en += 1
            else:
                pt += 1

        assert is_balanced(en, pt), "Language Weighted sampler is supposed to be balanced"

    def test_speaker_weighted_random_sampler(self):  # pylint: disable=no-self-use
        weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
            get_speaker_balancer_weights(train_samples), len(train_samples)
        )
        ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
        spk1, spk2 = 0, 0
        for index in ids:
            if train_samples[index]["speaker_name"] == "ljspeech-0":
                spk1 += 1
            else:
                spk2 += 1

        assert is_balanced(spk1, spk2), "Speaker Weighted sampler is supposed to be balanced"

    def test_perfect_sampler(self):  # pylint: disable=no-self-use
        classes = set()
        for item in train_samples:
            classes.add(item["speaker_name"])

        sampler = PerfectBatchSampler(
            train_samples,
            classes,
            batch_size=2 * 3,  # total batch size
            num_classes_in_batch=2,
            label_key="speaker_name",
            shuffle=False,
            drop_last=True,
        )
        batchs = functools.reduce(lambda a, b: a + b, [list(sampler) for i in range(100)])
        for batch in batchs:
            spk1, spk2 = 0, 0
            # for in each batch
            for index in batch:
                if train_samples[index]["speaker_name"] == "ljspeech-0":
                    spk1 += 1
                else:
                    spk2 += 1
            assert spk1 == spk2, "PerfectBatchSampler is supposed to be perfectly balanced"

    def test_perfect_sampler_shuffle(self):  # pylint: disable=no-self-use
        classes = set()
        for item in train_samples:
            classes.add(item["speaker_name"])

        sampler = PerfectBatchSampler(
            train_samples,
            classes,
            batch_size=2 * 3,  # total batch size
            num_classes_in_batch=2,
            label_key="speaker_name",
            shuffle=True,
            drop_last=False,
        )
        batchs = functools.reduce(lambda a, b: a + b, [list(sampler) for i in range(100)])
        for batch in batchs:
            spk1, spk2 = 0, 0
            # for in each batch
            for index in batch:
                if train_samples[index]["speaker_name"] == "ljspeech-0":
                    spk1 += 1
                else:
                    spk2 += 1
            assert spk1 == spk2, "PerfectBatchSampler is supposed to be perfectly balanced"

    def test_length_weighted_random_sampler(self):  # pylint: disable=no-self-use
        for _ in range(1000):
            # gerenate a lenght unbalanced dataset with random max/min audio lenght
            min_audio = random.randrange(1, 22050)
            max_audio = random.randrange(44100, 220500)
            for idx, item in enumerate(train_samples):
                # increase the diversity of durations
                random_increase = random.randrange(100, 1000)
                if idx < 5:
                    item["audio_length"] = min_audio + random_increase
                else:
                    item["audio_length"] = max_audio + random_increase

            weighted_sampler = torch.utils.data.sampler.WeightedRandomSampler(
                get_length_balancer_weights(train_samples, num_buckets=2), len(train_samples)
            )
            ids = functools.reduce(lambda a, b: a + b, [list(weighted_sampler) for i in range(100)])
            len1, len2 = 0, 0
            for index in ids:
                if train_samples[index]["audio_length"] < max_audio:
                    len1 += 1
                else:
                    len2 += 1
            assert is_balanced(len1, len2), "Length Weighted sampler is supposed to be balanced"

    def test_bucket_batch_sampler(self):
        bucket_size_multiplier = 2
        sampler = range(len(train_samples))
        sampler = BucketBatchSampler(
            sampler,
            data=train_samples,
            batch_size=7,
            drop_last=True,
            sort_key=lambda x: len(x["text"]),
            bucket_size_multiplier=bucket_size_multiplier,
        )

        # check if the samples are sorted by text lenght whuile bucketing
        min_text_len_in_bucket = 0
        bucket_items = []
        for batch_idx, batch in enumerate(list(sampler)):
            if (batch_idx + 1) % bucket_size_multiplier == 0:
                for bucket_item in bucket_items:
                    self.assertLessEqual(min_text_len_in_bucket, len(train_samples[bucket_item]["text"]))
                    min_text_len_in_bucket = len(train_samples[bucket_item]["text"])
                min_text_len_in_bucket = 0
                bucket_items = []
            else:
                bucket_items += batch

        # check sampler length
        self.assertEqual(len(sampler), len(train_samples) // 7)
