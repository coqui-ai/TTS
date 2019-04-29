import os
import unittest
import shutil
import numpy as np

from torch.utils.data import DataLoader
from utils.generic_utils import load_config
from utils.audio import AudioProcessor
from datasets import TTSDataset
from datasets.preprocess import ljspeech, tts_cache

file_path = os.path.dirname(os.path.realpath(__file__))
OUTPATH = os.path.join(file_path, "outputs/loader_tests/")
os.makedirs(OUTPATH, exist_ok=True)
c = load_config(os.path.join(file_path, 'test_config.json'))
ok_ljspeech = os.path.exists(c.data_path)

DATA_EXIST = True
CACHE_EXIST = True
if not os.path.exists(c.data_path_cache):
    CACHE_EXIST = False

if not os.path.exists(c.data_path):
    DATA_EXIST = False

print(" > Dynamic data loader test: {}".format(DATA_EXIST))
print(" > Cache data loader test: {}".format(CACHE_EXIST))

class TestTTSDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTTSDataset, self).__init__(*args, **kwargs)
        self.max_loader_iter = 4
        self.ap = AudioProcessor(**c.audio)

    def _create_dataloader(self, batch_size, r, bgs):
        dataset = TTSDataset.MyDataset(
            c.data_path,
            'metadata.csv',
            r,
            c.text_cleaner,
            preprocessor=ljspeech,
            ap=self.ap,
            batch_group_size=bgs,
            min_seq_len=c.min_seq_len,
            max_seq_len=float("inf"),
            use_phonemes=False)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=True,
            num_workers=c.num_loader_workers)
        return dataloader, dataset

    def test_loader(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(2, c.r, 0)

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
                assert linear_input.shape[2] == self.ap.num_freq
                assert mel_input.shape[0] == c.batch_size
                assert mel_input.shape[2] == c.audio['num_mels']
                # check normalization ranges
                if self.ap.symmetric_norm:
                    assert mel_input.max() <= self.ap.max_norm
                    assert mel_input.min() >= -self.ap.max_norm
                    assert mel_input.min() < 0
                else:
                    assert mel_input.max() <= self.ap.max_norm
                    assert mel_input.min() >= 0

    def test_batch_group_shuffle(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(2, c.r, 16)
            last_length = 0
            frames = dataset.items
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

                avg_length = mel_lengths.numpy().mean()
                assert avg_length >= last_length
            dataloader.dataset.sort_items()
            assert frames[0] != dataloader.dataset.items[0]

    def test_padding_and_spec(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(1, 1, 0)

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

                # check mel_spec consistency
                wav = self.ap.load_wav(item_idx[0])
                mel = self.ap.melspectrogram(wav)
                mel_dl = mel_input[0].cpu().numpy()
                assert (
                    abs(mel.T).astype("float32") - abs(mel_dl[:-1])).sum() == 0

                # check mel-spec correctness
                mel_spec = mel_input[0].cpu().numpy()
                wav = self.ap.inv_mel_spectrogram(mel_spec.T)
                self.ap.save_wav(wav, OUTPATH + '/mel_inv_dataloader.wav')
                shutil.copy(item_idx[0], OUTPATH + '/mel_target_dataloader.wav')

                # check linear-spec 
                linear_spec = linear_input[0].cpu().numpy()
                wav = self.ap.inv_spectrogram(linear_spec.T)
                self.ap.save_wav(wav, OUTPATH + '/linear_inv_dataloader.wav')
                shutil.copy(item_idx[0], OUTPATH + '/linear_target_dataloader.wav')

                # check the last time step to be zero padded
                assert linear_input[0, -1].sum() == 0
                assert linear_input[0, -2].sum() != 0
                assert mel_input[0, -1].sum() == 0
                assert mel_input[0, -2].sum() != 0
                assert stop_target[0, -1] == 1
                assert stop_target[0, -2] == 0
                assert stop_target.sum() == 1
                assert len(mel_lengths.shape) == 1
                assert mel_lengths[0] == linear_input[0].shape[0]
                assert mel_lengths[0] == mel_input[0].shape[0]

            # Test for batch size 2
            dataloader, dataset = self._create_dataloader(2, 1, 0)

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
                assert linear_input[idx, -1].sum() == 0
                assert linear_input[idx, -2].sum() != 0, linear_input
                assert mel_input[idx, -1].sum() == 0
                assert mel_input[idx, -2].sum() != 0, mel_input
                assert stop_target[idx, -1] == 1
                assert stop_target[idx, -2] == 0
                assert stop_target[idx].sum() == 1
                assert len(mel_lengths.shape) == 1
                assert mel_lengths[idx] == mel_input[idx].shape[0]
                assert mel_lengths[idx] == linear_input[idx].shape[0]

                # check the second itme in the batch
                assert linear_input[1 - idx, -1].sum() == 0
                assert mel_input[1 - idx, -1].sum() == 0
                assert stop_target[1 - idx, -1] == 1
                assert len(mel_lengths.shape) == 1

                # check batch conditions
                assert (linear_input * stop_target.unsqueeze(2)).sum() == 0
                assert (mel_input * stop_target.unsqueeze(2)).sum() == 0