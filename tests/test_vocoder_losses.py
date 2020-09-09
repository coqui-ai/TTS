import os

import torch
from tests import get_tests_input_path, get_tests_output_path, get_tests_path

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.layers.losses import MultiScaleSTFTLoss, STFTLoss, TorchSTFT

TESTS_PATH = get_tests_path()

OUT_PATH = os.path.join(get_tests_output_path(), "audio_tests")
os.makedirs(OUT_PATH, exist_ok=True)

WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")

C = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))
ap = AudioProcessor(**C.audio)


def test_torch_stft():
    torch_stft = TorchSTFT(ap.fft_size, ap.hop_length, ap.win_length)
    # librosa stft
    wav = ap.load_wav(WAV_FILE)
    M_librosa = abs(ap._stft(wav)) # pylint: disable=protected-access
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
    stft_loss = MultiScaleSTFTLoss([ap.fft_size//2, ap.fft_size, ap.fft_size*2],
                                   [ap.hop_length // 2, ap.hop_length, ap.hop_length * 2],
                                   [ap.win_length // 2, ap.win_length, ap.win_length * 2])
    wav = ap.load_wav(WAV_FILE)
    wav = torch.from_numpy(wav[None, :]).float()
    loss_m, loss_sc = stft_loss(wav, wav)
    assert loss_m + loss_sc == 0
    loss_m, loss_sc = stft_loss(wav, torch.rand_like(wav))
    assert loss_sc < 1.0
    assert loss_m + loss_sc > 0
