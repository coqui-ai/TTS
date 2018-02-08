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

        dataloader = DataLoader(dataset, batch_size=c.batch_size,
                                shuffle=True, collate_fn=dataset.collate_fn,
                                drop_last=True, num_workers=c.num_loader_workers)

        for i, data in enumerate(dataloader):
            if i == self.max_loader_iter:
                break
            text_input = data[0]
            text_lengths = data[1]
            mel_input = data[3]
            item_idx = data[4]

            neg_values = text_input[text_input < 0]
            check_count = len(neg_values)
            assert check_count == 0, \
                " !! Negative values in text_input: {}".format(check_count)




