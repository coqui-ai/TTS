import math
import os
import unittest
from dataclasses import dataclass

import librosa
import numpy as np
from coqpit import Coqpit

from tests import get_tests_input_path, get_tests_output_path, get_tests_path
from TTS.utils.audio import numpy_transforms as np_transforms

TESTS_PATH = get_tests_path()
OUT_PATH = os.path.join(get_tests_output_path(), "audio_tests")
WAV_FILE = os.path.join(get_tests_input_path(), "example_1.wav")

os.makedirs(OUT_PATH, exist_ok=True)


# pylint: disable=no-self-use


class TestNumpyTransforms(unittest.TestCase):
    def setUp(self) -> None:
        @dataclass
        class AudioConfig(Coqpit):
            sample_rate: int = 22050
            fft_size: int = 1024
            num_mels: int = 256
            mel_fmax: int = 1800
            mel_fmin: int = 0
            hop_length: int = 256
            win_length: int = 1024
            pitch_fmax: int = 640
            pitch_fmin: int = 1
            trim_db: int = -1
            min_silence_sec: float = 0.01
            gain: float = 1.0
            base: float = 10.0

        self.config = AudioConfig()
        self.sample_wav, _ = librosa.load(WAV_FILE, sr=self.config.sample_rate)

    def test_build_mel_basis(self):
        """Check if the mel basis is correctly built"""
        print(" > Testing mel basis building.")
        mel_basis = np_transforms.build_mel_basis(**self.config)
        self.assertEqual(mel_basis.shape, (self.config.num_mels, self.config.fft_size // 2 + 1))

    def test_millisec_to_length(self):
        """Check if the conversion from milliseconds to length is correct"""
        print(" > Testing millisec to length conversion.")
        win_len, hop_len = np_transforms.millisec_to_length(
            frame_length_ms=1000, frame_shift_ms=12.5, sample_rate=self.config.sample_rate
        )
        self.assertEqual(hop_len, int(12.5 / 1000.0 * self.config.sample_rate))
        self.assertEqual(win_len, self.config.sample_rate)

    def test_amplitude_db_conversion(self):
        di = np.random.rand(11)
        o1 = np_transforms.amp_to_db(x=di, gain=1.0, base=10)
        o2 = np_transforms.db_to_amp(x=o1, gain=1.0, base=10)
        np.testing.assert_almost_equal(di, o2, decimal=5)

    def test_preemphasis_deemphasis(self):
        di = np.random.rand(11)
        o1 = np_transforms.preemphasis(x=di, coeff=0.95)
        o2 = np_transforms.deemphasis(x=o1, coeff=0.95)
        np.testing.assert_almost_equal(di, o2, decimal=5)

    def test_spec_to_mel(self):
        mel_basis = np_transforms.build_mel_basis(**self.config)
        spec = np.random.rand(self.config.fft_size // 2 + 1, 20)  # [C, T]
        mel = np_transforms.spec_to_mel(spec=spec, mel_basis=mel_basis)
        self.assertEqual(mel.shape, (self.config.num_mels, 20))

    def mel_to_spec(self):
        mel_basis = np_transforms.build_mel_basis(**self.config)
        mel = np.random.rand(self.config.num_mels, 20)  # [C, T]
        spec = np_transforms.mel_to_spec(mel=mel, mel_basis=mel_basis)
        self.assertEqual(spec.shape, (self.config.fft_size // 2 + 1, 20))

    def test_wav_to_spec(self):
        spec = np_transforms.wav_to_spec(wav=self.sample_wav, **self.config)
        self.assertEqual(
            spec.shape, (self.config.fft_size // 2 + 1, math.ceil(self.sample_wav.shape[0] / self.config.hop_length))
        )

    def test_wav_to_mel(self):
        mel_basis = np_transforms.build_mel_basis(**self.config)
        mel = np_transforms.wav_to_mel(wav=self.sample_wav, mel_basis=mel_basis, **self.config)
        self.assertEqual(
            mel.shape, (self.config.num_mels, math.ceil(self.sample_wav.shape[0] / self.config.hop_length))
        )

    def test_compute_f0(self):
        pitch = np_transforms.compute_f0(x=self.sample_wav, **self.config)
        mel_basis = np_transforms.build_mel_basis(**self.config)
        mel = np_transforms.wav_to_mel(wav=self.sample_wav, mel_basis=mel_basis, **self.config)
        assert pitch.shape[0] == mel.shape[1]

    def test_load_wav(self):
        wav = np_transforms.load_wav(filename=WAV_FILE, resample=False, sample_rate=22050)
        wav_resample = np_transforms.load_wav(filename=WAV_FILE, resample=True, sample_rate=16000)
        self.assertEqual(wav.shape, (self.sample_wav.shape[0],))
        self.assertNotEqual(wav_resample.shape, (self.sample_wav.shape[0],))
