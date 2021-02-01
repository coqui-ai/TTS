import os

import numpy as np
from tests import get_tests_path, get_tests_input_path, get_tests_output_path
from torch.utils.data import DataLoader

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.datasets.gan_dataset import GANDataset
from TTS.vocoder.datasets.preprocess import load_wav_data

file_path = os.path.dirname(os.path.realpath(__file__))
OUTPATH = os.path.join(get_tests_output_path(), "loader_tests/")
os.makedirs(OUTPATH, exist_ok=True)

C = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))

test_data_path = os.path.join(get_tests_path(), "data/ljspeech/")
ok_ljspeech = os.path.exists(test_data_path)


def gan_dataset_case(batch_size, seq_len, hop_len, conv_pad, return_segments, use_noise_augment, use_cache, num_workers):
    ''' run dataloader with given parameters and check conditions '''
    ap = AudioProcessor(**C.audio)
    _, train_items = load_wav_data(test_data_path, 10)
    dataset = GANDataset(ap,
                         train_items,
                         seq_len=seq_len,
                         hop_len=hop_len,
                         pad_short=2000,
                         conv_pad=conv_pad,
                         return_segments=return_segments,
                         use_noise_augment=use_noise_augment,
                         use_cache=use_cache)
    loader = DataLoader(dataset=dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=num_workers,
                        pin_memory=True,
                        drop_last=True)

    max_iter = 10
    count_iter = 0

    # return random segments or return the whole audio
    if return_segments:
        for item1, _ in loader:
            feat1, wav1 = item1
            # feat2, wav2 = item2
            expected_feat_shape = (batch_size, ap.num_mels, seq_len // hop_len + conv_pad * 2)

            # check shapes
            assert np.all(feat1.shape == expected_feat_shape), f" [!] {feat1.shape} vs {expected_feat_shape}"
            assert (feat1.shape[2] - conv_pad * 2) * hop_len == wav1.shape[2]

            # check feature vs audio match
            if not use_noise_augment:
                for idx in range(batch_size):
                    audio = wav1[idx].squeeze()
                    feat = feat1[idx]
                    mel = ap.melspectrogram(audio)
                    # the first 2 and the last 2 frames are skipped due to the padding
                    # differences in stft
                    max_diff = abs((feat - mel[:, :feat1.shape[-1]])[:, 2:-2]).max()
                    assert max_diff <= 0, f' [!] {max_diff}'

            count_iter += 1
            # if count_iter == max_iter:
            #     break
    else:
        for item in loader:
            feat, wav = item
            expected_feat_shape = (batch_size, ap.num_mels, (wav.shape[-1] // hop_len) + (conv_pad * 2))
            assert np.all(feat.shape == expected_feat_shape), f" [!] {feat.shape} vs {expected_feat_shape}"
            assert (feat.shape[2] - conv_pad * 2) * hop_len == wav.shape[2]
            count_iter += 1
            if count_iter == max_iter:
                break


def test_parametrized_gan_dataset():
    ''' test dataloader with different parameters '''
    params = [
        [32, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, True, False, True, 0],
        [32, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, True, False, True, 4],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, True, True, True, 0],
        [1, C.audio['hop_length'], C.audio['hop_length'], 0, True, True, True, 0],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 2, True, True, True, 0],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, False, True, True, 0],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, True, False, True, 0],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, True, True, False, 0],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 0, False, False, False, 0],
    ]
    for param in params:
        print(param)
        gan_dataset_case(*param)
