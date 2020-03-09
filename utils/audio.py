import librosa
import soundfile as sf
import numpy as np
import scipy.io
import scipy.signal


class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 ref_level_db=None,
                 num_freq=None,
                 power=None,
                 preemphasis=0.0,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 sound_norm=False,
                 **_):

        print(" > Setting up Audio Processor...")

        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.num_freq = num_freq
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.sound_norm = sound_norm
        if hop_length is None:
            self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        else:
            self.hop_length = hop_length
            self.win_length = win_length
            self.n_fft = (self.num_freq - 1) * 2
        assert min_level_db != 0.0, " [!] min_level_db is 0"
        members = vars(self)
        for key, value in members.items():
            print(" | > {}:{}".format(key, value))

    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))

    def _linear_to_mel(self, spectrogram):
        _mel_basis = self._build_mel_basis()
        return np.dot(_mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec):
        inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        return np.maximum(1e-10, np.dot(inv_mel_basis, mel_spec))

    def _build_mel_basis(self, ):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        #pylint: disable=no-else-return
        if self.signal_norm:
            S_norm = ((S - self.min_level_db) / - self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def _denormalize(self, S):
        """denormalize values"""
        #pylint: disable=no-else-return
        S_denorm = S
        if self.signal_norm:
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm)
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db /
                            self.max_norm) + self.min_level_db
                return S_denorm
        else:
            return S

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        factor = self.frame_length_ms / self.frame_shift_ms
        assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return n_fft, hop_length, win_length

    def _amp_to_db(self, x):
        min_level = np.exp(self.min_level_db / 20 * np.log(10))
        return 20 * np.log10(np.maximum(min_level, x))

    @staticmethod
    def _db_to_amp(x):
        return np.power(10.0, x * 0.05)

    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D)) - self.ref_level_db
        return self._normalize(S)

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D))) - self.ref_level_db
        return self._normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self._denormalize(spectrogram)
        S = self._db_to_amp(S + self.ref_level_db)  # Convert back to linear
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def inv_mel_spectrogram(self, mel_spectrogram):
        '''Converts mel spectrogram to waveform using librosa'''
        D = self._denormalize(mel_spectrogram)
        S = self._db_to_amp(D + self.ref_level_db)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec):
        S = self._denormalize(linear_spec)
        S = self._db_to_amp(S + self.ref_level_db)
        S = self._linear_to_mel(np.abs(S))
        S = self._amp_to_db(S) - self.ref_level_db
        mel = self._normalize(S)
        return mel

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode='constant'
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def find_endpoint(self, wav, threshold_db=-40, min_silence_sec=0.8):
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(threshold_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x:x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """ Trim silent parts with a threshold and 0.01 sec margin """
        margin = int(self.sample_rate * 0.01)
        wav = wav[margin:-margin]
        return librosa.effects.trim(
            wav, top_db=self.trim_db, frame_length=self.win_length, hop_length=self.hop_length)[0]

    @staticmethod
    def mulaw_encode(wav, qc):
        mu = 2 ** qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1. + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(signal,)

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2 ** qc - 1
        x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
        return x

    def load_wav(self, filename, sr=None):
        if sr is None:
            x, sr = sf.read(filename)
        else:
            x, sr = librosa.load(filename, sr=sr)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(f' [!] File cannot be trimmed for silence - {filename}')
        assert self.sample_rate == sr, "%s vs %s"%(self.sample_rate, sr)
        if self.sound_norm:
            x = x / abs(x).max() * 0.9
        return x

    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x, bits):
        return (x + 1.) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        return 2 * x / (2**bits - 1) - 1
