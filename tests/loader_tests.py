import os
import unittest
import numpy as np

from torch.utils.data import DataLoader
from TTS.utils.generic_utils import load_config
from TTS.datasets.LJSpeech import LJSpeechDataset


file_path = os.path.dirname(os.path.realpath(__file__))
c = load_config(os.path.join(file_path, 'test_config.json'))


class TestDataset(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super(TestDataset, self).__init__(*args, **kwargs)
        self.max_loader_iter = 4

    def test_loader(self):
        dataset = LJSpeechDataset(os.path.join(c.data_path, 'metadata.csv'),
                                  os.path.join(c.data_path, 'wavs'),
                                  c.r,
                                  c.sample_rate,
                                  c.text_cleaner,
                                  c.num_mels,
                                  c.min_level_db,
                                  c.frame_shift_ms,
                                  c.frame_length_ms,
                                  c.preemphasis,
                                  c.ref_level_db,
                                  c.num_freq,
                                  c.power
                                  )

        dataloader = DataLoader(dataset, batch_size=2,
                                shuffle=True, collate_fn=dataset.collate_fn,
                                drop_last=True, num_workers=c.num_loader_workers)

        for i, data in enumerate(dataloader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            neg_values = text_input[text_input < 0]
            check_count = len(neg_values)
            assert check_count == 0, \
                " !! Negative values in text_input: {}".format(check_count)
            # TODO: more assertion here
            assert linear_input.shape[0] == c.batch_size
            assert mel_input.shape[0] == c.batch_size
            assert mel_input.shape[2] == c.num_mels

    def test_padding(self):
        dataset = LJSpeechDataset(os.path.join(c.data_path, 'metadata.csv'),
                                  os.path.join(c.data_path, 'wavs'),
                                  1,
                                  c.sample_rate,
                                  c.text_cleaner,
                                  c.num_mels,
                                  c.min_level_db,
                                  c.frame_shift_ms,
                                  c.frame_length_ms,
                                  c.preemphasis,
                                  c.ref_level_db,
                                  c.num_freq,
                                  c.power
                                  )

        # Test for batch size 1
        dataloader = DataLoader(dataset, batch_size=1,
                                shuffle=False, collate_fn=dataset.collate_fn,
                                drop_last=True, num_workers=c.num_loader_workers)

        for i, data in enumerate(dataloader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            # check the last time step to be zero padded
            assert mel_input[0, -1].sum() == 0
            assert mel_input[0, -2].sum() != 0
            assert linear_input[0, -1].sum() == 0
            assert linear_input[0, -2].sum() != 0
            assert stop_target[0, -1] == 1
            assert stop_target[0, -2] == 0
            assert stop_target.sum() == 1
            assert len(mel_lengths.shape) == 1
            assert mel_lengths[0] == mel_input[0].shape[0]

        # Test for batch size 2
        dataloader = DataLoader(dataset, batch_size=2,
                                shuffle=False, collate_fn=dataset.collate_fn,
                                drop_last=False, num_workers=c.num_loader_workers)

        for i, data in enumerate(dataloader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]
            mel_lengths = data[4]
            stop_target = data[5]
            item_idx = data[6]

            if mel_lengths[0] > mel_lengths[1]:
                idx = 0
            else:
                idx = 1

            # check the first item in the batch
            assert mel_input[idx, -1].sum() == 0
            assert mel_input[idx, -2].sum() != 0, mel_input
            assert linear_input[idx, -1].sum() == 0
            assert linear_input[idx, -2].sum() != 0
            assert stop_target[idx, -1] == 1
            assert stop_target[idx, -2] == 0
            assert stop_target[idx].sum() == 1
            assert len(mel_lengths.shape) == 1
            assert mel_lengths[idx] == mel_input[idx].shape[0]

            # check the second itme in the batch
            assert mel_input[1-idx, -1].sum() == 0
            assert linear_input[1-idx, -1].sum() == 0
            assert stop_target[1-idx, -1] == 1
            assert len(mel_lengths.shape) == 1

            # check batch conditions
            assert (mel_input * stop_target.unsqueeze(2)).sum() == 0
            assert (linear_input * stop_target.unsqueeze(2)).sum() == 0
