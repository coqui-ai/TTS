import os

import torch

from tests import get_tests_input_path, get_tests_output_path, get_tests_path
from TTS.config import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.utils.audio.numpy_transforms import stft
from TTS.vocoder.layers.losses import MelganFeatureLoss, MultiScaleSTFTLoss, STFTLoss, TorchSTFT

TESTS_PATH = get_tests_path()

OUT_PATH = os.path.join(get_tests_output_path(), "audio_tests")
os.makedirs(OUT_PATH, exist_ok=True)

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")

ap = AudioProcessor(**BaseAudioConfig().to_dict())


def test_torch_stft():
    torch_stft = TorchSTFT(ap.fft_size, ap.hop_length, ap.win_length)
    # librosa stft
    wav = ap.load_wav(WAV_FILE)
    M_librosa = abs(stft(y=wav, fft_size=ap.fft_size, hop_length=ap.hop_length, win_length=ap.win_length))
    # torch stft
    wav = torch.from_numpy(wav[None, :]).float()
    M_torch = torch_stft(wav)
    # check the difference b/w librosa and torch outputs
    assert (M_librosa - M_torch[0].data.numpy()).max() < 1e-5


def test_stft_loss():
    stft_loss = STFTLoss(ap.fft_size, ap.hop_length, ap.win_length)
    wav = ap.load_wav(WAV_FILE)
    wav = torch.from_numpy(wav[None, :]).float()
    loss_m, loss_sc = stft_loss(wav, wav)
    assert loss_m + loss_sc == 0
    loss_m, loss_sc = stft_loss(wav, torch.rand_like(wav))
    assert loss_sc < 1.0
    assert loss_m + loss_sc > 0


def test_multiscale_stft_loss():
    stft_loss = MultiScaleSTFTLoss(
        [ap.fft_size // 2, ap.fft_size, ap.fft_size * 2],
        [ap.hop_length // 2, ap.hop_length, ap.hop_length * 2],
        [ap.win_length // 2, ap.win_length, ap.win_length * 2],
    )
    wav = ap.load_wav(WAV_FILE)
    wav = torch.from_numpy(wav[None, :]).float()
    loss_m, loss_sc = stft_loss(wav, wav)
    assert loss_m + loss_sc == 0
    loss_m, loss_sc = stft_loss(wav, torch.rand_like(wav))
    assert loss_sc < 1.0
    assert loss_m + loss_sc > 0


def test_melgan_feature_loss():
    feats_real = []
    feats_fake = []

    # if all the features are different.
    for _ in range(5):  # different scales
        scale_feats_real = []
        scale_feats_fake = []
        for _ in range(4):  # different layers
            scale_feats_real.append(torch.rand([3, 5, 7]))
            scale_feats_fake.append(torch.rand([3, 5, 7]))
        feats_real.append(scale_feats_real)
        feats_fake.append(scale_feats_fake)

    loss_func = MelganFeatureLoss()
    loss = loss_func(feats_fake, feats_real)
    assert loss.item() <= 1.0

    feats_real = []
    feats_fake = []

    # if all the features are the same
    for _ in range(5):  # different scales
        scale_feats_real = []
        scale_feats_fake = []
        for _ in range(4):  # different layers
            tensor = torch.rand([3, 5, 7])
            scale_feats_real.append(tensor)
            scale_feats_fake.append(tensor)
        feats_real.append(scale_feats_real)
        feats_fake.append(scale_feats_fake)

    loss_func = MelganFeatureLoss()
    loss = loss_func(feats_fake, feats_real)
    assert loss.item() == 0
