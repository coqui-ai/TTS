import os
import shutil
import unittest

import numpy as np
import torch
from torch.utils.data import DataLoader

from tests import get_tests_data_path, get_tests_output_path
from TTS.tts.configs.shared_configs import BaseDatasetConfig, BaseTTSConfig
from TTS.tts.datasets import TTSDataset, load_tts_samples
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# pylint: disable=unused-variable

OUTPATH = os.path.join(get_tests_output_path(), "loader_tests/")
os.makedirs(OUTPATH, exist_ok=True)

# create a dummy config for testing data loaders.
c = BaseTTSConfig(text_cleaner="english_cleaners", num_loader_workers=0, batch_size=2, use_noise_augment=False)
c.r = 5
c.data_path = os.path.join(get_tests_data_path(), "ljspeech/")
ok_ljspeech = os.path.exists(c.data_path)

dataset_config = BaseDatasetConfig(
    formatter="ljspeech_test",  # ljspeech_test to multi-speaker
    meta_file_train="metadata.csv",
    meta_file_val=None,
    path=c.data_path,
    language="en",
)

DATA_EXIST = True
if not os.path.exists(c.data_path):
    DATA_EXIST = False

print(" > Dynamic data loader test: {}".format(DATA_EXIST))


class TestTTSDataset(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_loader_iter = 4
        self.ap = AudioProcessor(**c.audio)

    def _create_dataloader(self, batch_size, r, bgs, start_by_longest=False):
        # load dataset
        meta_data_train, meta_data_eval = load_tts_samples(dataset_config, eval_split=True, eval_split_size=0.2)
        items = meta_data_train + meta_data_eval

        tokenizer, _ = TTSTokenizer.init_from_config(c)
        dataset = TTSDataset(
            outputs_per_step=r,
            compute_linear_spec=True,
            return_wav=True,
            tokenizer=tokenizer,
            ap=self.ap,
            samples=items,
            batch_group_size=bgs,
            min_text_len=c.min_text_len,
            max_text_len=c.max_text_len,
            min_audio_len=c.min_audio_len,
            max_audio_len=c.max_audio_len,
            start_by_longest=start_by_longest,
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
            dataloader, dataset = self._create_dataloader(1, 1, 0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                text_input = data["token_id"]
                _ = data["token_id_lengths"]
                speaker_name = data["speaker_names"]
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                _ = data["stop_targets"]
                _ = data["item_idxs"]
                wavs = data["waveform"]

                neg_values = text_input[text_input < 0]
                check_count = len(neg_values)

                # check basic conditions
                self.assertEqual(check_count, 0)
                self.assertEqual(linear_input.shape[0], mel_input.shape[0], c.batch_size)
                self.assertEqual(linear_input.shape[2], self.ap.fft_size // 2 + 1)
                self.assertEqual(mel_input.shape[2], c.audio["num_mels"])
                self.assertEqual(wavs.shape[1], mel_input.shape[1] * c.audio.hop_length)
                self.assertIsInstance(speaker_name[0], str)

                # make sure that the computed mels and the waveform match and correctly computed
                mel_new = self.ap.melspectrogram(wavs[0].squeeze().numpy())
                # remove padding in mel-spectrogram
                mel_dataloader = mel_input[0].T.numpy()[:, : mel_lengths[0]]
                # guarantee that both mel-spectrograms have the same size and that we will remove waveform padding
                mel_new = mel_new[:, : mel_lengths[0]]
                ignore_seg = -(1 + c.audio.win_length // c.audio.hop_length)
                mel_diff = (mel_new[:, : mel_input.shape[1]] - mel_input[0].T.numpy())[:, 0:ignore_seg]
                self.assertLess(abs(mel_diff.sum()), 1e-5)

                # check normalization ranges
                if self.ap.symmetric_norm:
                    self.assertLessEqual(mel_input.max(), self.ap.max_norm)
                    self.assertGreaterEqual(
                        mel_input.min(), -self.ap.max_norm  # pylint: disable=invalid-unary-operand-type
                    )
                    self.assertLess(mel_input.min(), 0)
                else:
                    self.assertLessEqual(mel_input.max(), self.ap.max_norm)
                    self.assertGreaterEqual(mel_input.min(), 0)

    def test_batch_group_shuffle(self):
        if ok_ljspeech:
            dataloader, dataset = self._create_dataloader(2, c.r, 16)
            last_length = 0
            frames = dataset.samples
            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                mel_lengths = data["mel_lengths"]
                avg_length = mel_lengths.numpy().mean()
            dataloader.dataset.preprocess_samples()
            is_items_reordered = False
            for idx, item in enumerate(dataloader.dataset.samples):
                if item != frames[idx]:
                    is_items_reordered = True
                    break
            self.assertGreaterEqual(avg_length, last_length)
            self.assertTrue(is_items_reordered)

    def test_start_by_longest(self):
        """Test start_by_longest option.

        Ther first item of the fist batch must be longer than all the other items.
        """
        if ok_ljspeech:
            dataloader, _ = self._create_dataloader(2, c.r, 0, True)
            dataloader.dataset.preprocess_samples()
            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                mel_lengths = data["mel_lengths"]
                if i == 0:
                    max_len = mel_lengths[0]
                print(mel_lengths)
                self.assertTrue(all(max_len >= mel_lengths))

    def test_padding_and_spectrograms(self):
        def check_conditions(idx, linear_input, mel_input, stop_target, mel_lengths):
            self.assertNotEqual(linear_input[idx, -1].sum(), 0)  # check padding
            self.assertNotEqual(linear_input[idx, -2].sum(), 0)
            self.assertNotEqual(mel_input[idx, -1].sum(), 0)
            self.assertNotEqual(mel_input[idx, -2].sum(), 0)
            self.assertEqual(stop_target[idx, -1], 1)
            self.assertEqual(stop_target[idx, -2], 0)
            self.assertEqual(stop_target[idx].sum(), 1)
            self.assertEqual(len(mel_lengths.shape), 1)
            self.assertEqual(mel_lengths[idx], linear_input[idx].shape[0])
            self.assertEqual(mel_lengths[idx], mel_input[idx].shape[0])

        if ok_ljspeech:
            dataloader, _ = self._create_dataloader(1, 1, 0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
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
                self.assertLess(abs(mel.T - mel_dl).max(), 1e-5)

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

                # check the outputs
                check_conditions(0, linear_input, mel_input, stop_target, mel_lengths)

            # Test for batch size 2
            dataloader, _ = self._create_dataloader(2, 1, 0)

            for i, data in enumerate(dataloader):
                if i == self.max_loader_iter:
                    break
                linear_input = data["linear"]
                mel_input = data["mel"]
                mel_lengths = data["mel_lengths"]
                stop_target = data["stop_targets"]
                item_idx = data["item_idxs"]

                # set id to the longest sequence in the batch
                if mel_lengths[0] > mel_lengths[1]:
                    idx = 0
                else:
                    idx = 1

                # check the longer item in the batch
                check_conditions(idx, linear_input, mel_input, stop_target, mel_lengths)

                # check the other item in the batch
                self.assertEqual(linear_input[1 - idx, -1].sum(), 0)
                self.assertEqual(mel_input[1 - idx, -1].sum(), 0)
                self.assertEqual(stop_target[1, mel_lengths[1] - 1], 1)
                self.assertEqual(stop_target[1, mel_lengths[1] :].sum(), stop_target.shape[1] - mel_lengths[1])
                self.assertEqual(len(mel_lengths.shape), 1)

                # check batch zero-frame conditions (zero-frame disabled)
                # assert (linear_input * stop_target.unsqueeze(2)).sum() == 0
                # assert (mel_input * stop_target.unsqueeze(2)).sum() == 0
