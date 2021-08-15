import os
import shutil
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests import get_tests_output_path
from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets import TTSDataset
from TTS.tts.datasets.formatters import ljspeech
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

OUTPATH = os.path.join(get_tests_output_path(), "loader_tests/")
os.makedirs(OUTPATH, exist_ok=True)

# create a dummy config for testing data loaders.
c = BaseTTSConfig(text_cleaner="english_cleaners", num_loader_workers=0, batch_size=2)
c.r = 5
c.data_path = "tests/data/ljspeech/"
ok_ljspeech = os.path.exists(c.data_path)

DATA_EXIST = True
if not os.path.exists(c.data_path):
    DATA_EXIST = False

print(" > Dynamic data loader test: {}".format(DATA_EXIST))


class TestTTSDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_loader_iter = 4
        self.ap = AudioProcessor(**c.audio)

    def _create_dataloader(self, batch_size, r, bgs):
        items = ljspeech(c.data_path, "metadata.csv")

        # add a default language because now the TTSDataset expect a language
        language = ""
        items = [[*item, language] for item in items]

        dataset = TTSDataset(
            r,
            c.text_cleaner,
            compute_linear_spec=True,
            return_wav=True,
            ap=self.ap,
            meta_data=items,
            characters=c.characters,
            batch_group_size=bgs,
            min_seq_len=c.min_seq_len,
            max_seq_len=float("inf"),
            use_phonemes=False,
        )
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=True,
            num_workers=c.num_loader_workers,
        )
        return dataloader, dataset

    def test_loader(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(2, c.r, 0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                text_input = data["text"]
                text_lengths = data["text_lengths"]
                speaker_name = data["speaker_names"]
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                stop_target = data["stop_targets"]
                item_idx = data["item_idxs"]
                wavs = data["waveform"]

                neg_values = text_input[text_input < 0]
                check_count = len(neg_values)
                assert check_count == 0, " !! Negative values in text_input: {}".format(check_count)
                assert isinstance(speaker_name[0], str)
                assert linear_input.shape[0] == c.batch_size
                assert linear_input.shape[2] == self.ap.fft_size // 2 + 1
                assert mel_input.shape[0] == c.batch_size
                assert mel_input.shape[2] == c.audio["num_mels"]
                assert (
                    wavs.shape[1] == mel_input.shape[1] * c.audio.hop_length
                ), f"wavs.shape: {wavs.shape[1]}, mel_input.shape: {mel_input.shape[1] * c.audio.hop_length}"

                # make sure that the computed mels and the waveform match and correctly computed
                mel_new = self.ap.melspectrogram(wavs[0].squeeze().numpy())
                ignore_seg = -(1 + c.audio.win_length // c.audio.hop_length)
                mel_diff = (mel_new[:, : mel_input.shape[1]] - mel_input[0].T.numpy())[:, 0:ignore_seg]
                assert abs(mel_diff.sum()) < 1e-5

                # check normalization ranges
                if self.ap.symmetric_norm:
                    assert mel_input.max() <= self.ap.max_norm
                    assert mel_input.min() >= -self.ap.max_norm  # pylint: disable=invalid-unary-operand-type
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
                text_input = data["text"]
                text_lengths = data["text_lengths"]
                speaker_name = data["speaker_names"]
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                stop_target = data["stop_targets"]
                item_idx = data["item_idxs"]

                avg_length = mel_lengths.numpy().mean()
                assert avg_length >= last_length
            dataloader.dataset.sort_and_filter_items()
            is_items_reordered = False
            for idx, item in enumerate(dataloader.dataset.items):
                if item != frames[idx]:
                    is_items_reordered = True
                    break
            assert is_items_reordered

    def test_padding_and_spec(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(1, 1, 0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                text_input = data["text"]
                text_lengths = data["text_lengths"]
                speaker_name = data["speaker_names"]
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                stop_target = data["stop_targets"]
                item_idx = data["item_idxs"]

                # check mel_spec consistency
                wav = np.asarray(self.ap.load_wav(item_idx[0]), dtype=np.float32)
                mel = self.ap.melspectrogram(wav).astype("float32")
                mel = torch.FloatTensor(mel).contiguous()
                mel_dl = mel_input[0]
                # NOTE: Below needs to check == 0 but due to an unknown reason
                # there is a slight difference between two matrices.
                # TODO: Check this assert cond more in detail.
                assert abs(mel.T - mel_dl).max() < 1e-5, abs(mel.T - mel_dl).max()

                # check mel-spec correctness
                mel_spec = mel_input[0].cpu().numpy()
                wav = self.ap.inv_melspectrogram(mel_spec.T)
                self.ap.save_wav(wav, OUTPATH + "/mel_inv_dataloader.wav")
                shutil.copy(item_idx[0], OUTPATH + "/mel_target_dataloader.wav")

                # check linear-spec
                linear_spec = linear_input[0].cpu().numpy()
                wav = self.ap.inv_spectrogram(linear_spec.T)
                self.ap.save_wav(wav, OUTPATH + "/linear_inv_dataloader.wav")
                shutil.copy(item_idx[0], OUTPATH + "/linear_target_dataloader.wav")

                # check the last time step to be zero padded
                assert linear_input[0, -1].sum() != 0
                assert linear_input[0, -2].sum() != 0
                assert mel_input[0, -1].sum() != 0
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
                text_input = data["text"]
                text_lengths = data["text_lengths"]
                speaker_name = data["speaker_names"]
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                stop_target = data["stop_targets"]
                item_idx = data["item_idxs"]

                if mel_lengths[0] > mel_lengths[1]:
                    idx = 0
                else:
                    idx = 1

                # check the first item in the batch
                assert linear_input[idx, -1].sum() != 0
                assert linear_input[idx, -2].sum() != 0, linear_input
                assert mel_input[idx, -1].sum() != 0
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
                assert stop_target[1, mel_lengths[1] - 1] == 1
                assert stop_target[1, mel_lengths[1] :].sum() == stop_target.shape[1] - mel_lengths[1]
                assert len(mel_lengths.shape) == 1

                # check batch zero-frame conditions (zero-frame disabled)
                # assert (linear_input * stop_target.unsqueeze(2)).sum() == 0
                # assert (mel_input * stop_target.unsqueeze(2)).sum() == 0
