import os
import shutil

import numpy as np
from tests import get_tests_path, get_tests_input_path, get_tests_output_path
from torch.utils.data import DataLoader

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.datasets.wavernn_dataset import WaveRNNDataset
from TTS.vocoder.datasets.preprocess import load_wav_feat_data, preprocess_wav_files

file_path = os.path.dirname(os.path.realpath(__file__))
OUTPATH = os.path.join(get_tests_output_path(), "loader_tests/")
os.makedirs(OUTPATH, exist_ok=True)

C = load_config(os.path.join(get_tests_input_path(),
                             "test_vocoder_wavernn_config.json"))

test_data_path = os.path.join(get_tests_path(), "data/ljspeech/")
test_mel_feat_path = os.path.join(test_data_path, "mel")
test_quant_feat_path = os.path.join(test_data_path, "quant")
ok_ljspeech = os.path.exists(test_data_path)


def wavernn_dataset_case(batch_size, seq_len, hop_len, pad, mode, mulaw, num_workers):
    """ run dataloader with given parameters and check conditions """
    ap = AudioProcessor(**C.audio)

    C.batch_size = batch_size
    C.mode = mode
    C.seq_len = seq_len
    C.data_path = test_data_path

    preprocess_wav_files(test_data_path, C, ap)
    _, train_items = load_wav_feat_data(
        test_data_path, test_mel_feat_path, 5)

    dataset = WaveRNNDataset(ap=ap,
                             items=train_items,
                             seq_len=seq_len,
                             hop_len=hop_len,
                             pad=pad,
                             mode=mode,
                             mulaw=mulaw
                             )
    # sampler = DistributedSampler(dataset) if num_gpus > 1 else None
    loader = DataLoader(dataset,
                        shuffle=True,
                        collate_fn=dataset.collate,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        pin_memory=True,
                        )

    max_iter = 10
    count_iter = 0

    try:
        for data in loader:
            x_input, mels, _ = data
            expected_feat_shape = (ap.num_mels,
                                   (x_input.shape[-1] // hop_len) + (pad * 2))
            assert np.all(
                mels.shape[1:] == expected_feat_shape), f" [!] {mels.shape} vs {expected_feat_shape}"

            assert (mels.shape[2] - pad * 2) * hop_len == x_input.shape[1]
            count_iter += 1
            if count_iter == max_iter:
                break
    # except AssertionError:
    #     shutil.rmtree(test_mel_feat_path)
    #     shutil.rmtree(test_quant_feat_path)
    finally:
        shutil.rmtree(test_mel_feat_path)
        shutil.rmtree(test_quant_feat_path)


def test_parametrized_wavernn_dataset():
    ''' test dataloader with different parameters '''
    params = [
        [16, C.audio['hop_length'] * 10, C.audio['hop_length'], 2, 10, True, 0],
        [16, C.audio['hop_length'] * 10, C.audio['hop_length'], 2, "mold", False, 4],
        [1, C.audio['hop_length'] * 10, C.audio['hop_length'], 2, 9, False, 0],
        [1, C.audio['hop_length'], C.audio['hop_length'], 2, 10, True, 0],
        [1, C.audio['hop_length'], C.audio['hop_length'], 2, "mold", False, 0],
        [1, C.audio['hop_length'] * 5, C.audio['hop_length'], 4, 10, False, 2],
        [1, C.audio['hop_length'] * 5, C.audio['hop_length'], 2, "mold", False, 0],
    ]
    for param in params:
        print(param)
        wavernn_dataset_case(*param)
