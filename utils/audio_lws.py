import os
import sys
import librosa
import pickle
import copy
import numpy as np
from scipy import signal
import lws

_mel_basis = None


class AudioProcessor(object):
    def __init__(
            self,
            sample_rate,
            num_mels,
            min_level_db,
            frame_shift_ms,
            frame_length_ms,
            ref_level_db,
            num_freq,
            power,
            preemphasis,
            min_mel_freq,
            max_mel_freq,
            griffin_lim_iters=None,
    ):
        print(" > Setting up Audio Processor...")
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.min_mel_freq = min_mel_freq
        self.max_mel_freq = max_mel_freq
        self.griffin_lim_iters = griffin_lim_iters
        self.preemphasis = preemphasis
        self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        if preemphasis == 0:
            print(" | > Preemphasis is deactive.")

    def save_wav(self, wav, path):
        wav *= 32767 / max(0.01, np.max(np.abs(wav)))
        librosa.output.write_wav(
            path, wav.astype(np.int16), self.sample_rate)

    def _stft_parameters(self, ):
        n_fft = int((self.num_freq - 1) * 2)
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(self.frame_length_ms / 1000.0 * self.sample_rate)
        if n_fft % hop_length != 0:
            hop_length = n_fft / 8
            print(" | > hop_length is set to default ({}).".format(hop_length))
        if n_fft % win_length != 0:
            win_length = n_fft / 2
            print(" | > win_length is set to default ({}).".format(win_length))
        print(" | > fft size: {}, hop length: {}, win length: {}".format(
            n_fft, hop_length, win_length))
        return int(n_fft), int(hop_length), int(win_length)

    def _lws_processor(self):
        try:
            return lws.lws(
                self.win_length,
                self.hop_length,
                fftsize=self.n_fft,
                mode="speech")
        except:
            raise RuntimeError(
                " !! WindowLength({}) is not multiple of HopLength({}).".
                format(self.win_length, self.hop_length))

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    def _db_to_amp(self, x):
        return np.power(10.0, x * 0.05)

    def _normalize(self, S):
        return np.clip((S - self.min_level_db) / -self.min_level_db, 0, 1)

    def _denormalize(self, S):
        return (np.clip(S, 0, 1) * -self.min_level_db) + self.min_level_db

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" !! Preemphasis is applied with factor 0.0. ")
        return signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        f = open(os.devnull, 'w')
        old_out = sys.stdout
        sys.stdout = f
        if self.preemphasis:
            D = self._lws_processor().stft(self.apply_preemphasis(y)).T
        else:
            D = self._lws_processor().stft(y).T
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        sys.stdout = old_out
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        '''Converts spectrogram to waveform using librosa'''
        f = open(os.devnull, 'w')
        old_out = sys.stdout
        sys.stdout = f
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        processor = self._lws_processor()
        D = processor.run_lws(S.astype(np.float64).T**self.power)
        y = processor.istft(D).astype(np.float32)
        # Reconstruct phase
        sys.stdout = old_out
        if self.preemphasis:
            return self.apply_inv_preemphasis(y)
        return y

    def _linear_to_mel(self, spectrogram):
        global _mel_basis
        if _mel_basis is None:
            _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _build_mel_basis(self, ):
        return librosa.filters.mel(
            self.sample_rate, self.n_fft, n_mels=self.num_mels)


#                                    fmin=self.min_mel_freq, fmax=self.max_mel_freq)

    def melspectrogram(self, y):
        f = open(os.devnull, 'w')
        old_out = sys.stdout
        sys.stdout = f
        if self.preemphasis:
            D = self._lws_processor().stft(self.apply_preemphasis(y)).T
        else:
            D = self._lws_processor().stft(y).T
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        sys.stdout = old_out
        return self._normalize(S)
