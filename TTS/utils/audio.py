import librosa
import soundfile as sf
import numpy as np
import scipy.io.wavfile
import scipy.signal
# import pyworld as pw

from TTS.tts.utils.data import StandardScaler

#pylint: disable=too-many-public-methods
class AudioProcessor(object):
    """Audio Processor for TTS used by all the data pipelines.

    Note:
        All the class arguments are set to default values to enable a flexible initialization
        of the class with the model config. They are not meaningful for all the arguments.

    Args:
        sample_rate (int, optional): target audio sampling rate. Defaults to None.
        resample (bool, optional): enable/disable resampling of the audio clips when the target sampling rate does not match the original sampling rate. Defaults to False.
        num_mels (int, optional): number of melspectrogram dimensions. Defaults to None.
        min_level_db (int, optional): minimum db threshold for the computed melspectrograms. Defaults to None.
        frame_shift_ms (int, optional): milliseconds of frames between STFT columns. Defaults to None.
        frame_length_ms (int, optional): milliseconds of STFT window length. Defaults to None.
        hop_length (int, optional): number of frames between STFT columns. Used if ```frame_shift_ms``` is None. Defaults to None.
        win_length (int, optional): STFT window length. Used if ```frame_length_ms``` is None. Defaults to None.
        ref_level_db (int, optional): reference DB level to avoid background noise. In general <20DB corresponds to the air noise. Defaults to None.
        fft_size (int, optional): FFT window size for STFT. Defaults to 1024.
        power (int, optional): Exponent value applied to the spectrogram before GriffinLim. Defaults to None.
        preemphasis (float, optional): Preemphasis coefficient. Preemphasis is disabled if == 0.0. Defaults to 0.0.
        signal_norm (bool, optional): enable/disable signal normalization. Defaults to None.
        symmetric_norm (bool, optional): enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else [0, k], Defaults to None.
        max_norm (float, optional): ```k``` defining the normalization range. Defaults to None.
        mel_fmin (int, optional): minimum filter frequency for computing melspectrograms. Defaults to None.
        mel_fmax (int, optional): maximum filter frequency for computing melspectrograms.. Defaults to None.
        spec_gain (int, optional): gain applied when converting amplitude to DB. Defaults to 20.
        stft_pad_mode (str, optional): Padding mode for STFT. Defaults to 'reflect'.
        clip_norm (bool, optional): enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.
        griffin_lim_iters (int, optional): Number of GriffinLim iterations. Defaults to None.
        do_trim_silence (bool, optional): enable/disable silence trimming when loading the audio signal. Defaults to False.
        trim_db (int, optional): DB threshold used for silence trimming. Defaults to 60.
        do_sound_norm (bool, optional): enable/disable signal normalization. Defaults to False.
        stats_path (str, optional): Path to the computed stats file. Defaults to None.
        verbose (bool, optional): enable/disable logging. Defaults to True.
    """
    def __init__(self,
                 sample_rate=None,
                 resample=False,
                 num_mels=None,
                 min_level_db=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 ref_level_db=None,
                 fft_size=1024,
                 power=None,
                 preemphasis=0.0,
                 signal_norm=None,
                 symmetric_norm=None,
                 max_norm=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 spec_gain=20,
                 stft_pad_mode='reflect',
                 clip_norm=True,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 do_sound_norm=False,
                 stats_path=None,
                 verbose=True,
                 **_):

        # setup class attributed
        self.sample_rate = sample_rate
        self.resample = resample
        self.num_mels = num_mels
        self.min_level_db = min_level_db or 0
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.ref_level_db = ref_level_db
        self.fft_size = fft_size
        self.power = power
        self.preemphasis = preemphasis
        self.griffin_lim_iters = griffin_lim_iters
        self.signal_norm = signal_norm
        self.symmetric_norm = symmetric_norm
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = stft_pad_mode
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.stats_path = stats_path
        # setup stft parameters
        if hop_length is None:
            # compute stft parameters from given time values
            self.hop_length, self.win_length = self._stft_parameters()
        else:
            # use stft parameters from config file
            self.hop_length = hop_length
            self.win_length = win_length
        assert min_level_db != 0.0, " [!] min_level_db is 0"
        assert self.win_length <= self.fft_size, " [!] win_length cannot be larger than fft_size"
        members = vars(self)
        if verbose:
            print(" > Setting up Audio Processor...")
            for key, value in members.items():
                print(" | > {}:{}".format(key, value))
        # create spectrogram utils
        self.mel_basis = self._build_mel_basis()
        self.inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        # setup scaler
        if stats_path:
            mel_mean, mel_std, linear_mean, linear_std, _ = self.load_stats(stats_path)
            self.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)
            self.signal_norm = True
            self.max_norm = None
            self.clip_norm = None
            self.symmetric_norm = None

    ### setting up the parameters ###
    def _build_mel_basis(self, ):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.fft_size,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        factor = self.frame_length_ms / self.frame_shift_ms
        assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return hop_length, win_length

    ### normalization ###
    def normalize(self, S):
        """Put values in [0, self.max_norm] or [-self.max_norm, self.max_norm]"""
        #pylint: disable=no-else-return
        S = S.copy()
        if self.signal_norm:
            # mean-var scaling
            if hasattr(self, 'mel_scaler'):
                if S.shape[0] == self.num_mels:
                    return self.mel_scaler.transform(S.T).T
                elif S.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.transform(S.T).T
                else:
                    raise RuntimeError(' [!] Mean-Var stats does not match the given feature dimensions.')
            # range normalization
            S -= self.ref_level_db  # discard certain range of DB assuming it is air noise
            S_norm = ((S - self.min_level_db) / (-self.min_level_db))
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, -self.max_norm, self.max_norm)  # pylint: disable=invalid-unary-operand-type
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def denormalize(self, S):
        """denormalize values"""
        #pylint: disable=no-else-return
        S_denorm = S.copy()
        if self.signal_norm:
            # mean-var scaling
            if hasattr(self, 'mel_scaler'):
                if S_denorm.shape[0] == self.num_mels:
                    return self.mel_scaler.inverse_transform(S_denorm.T).T
                elif S_denorm.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.inverse_transform(S_denorm.T).T
                else:
                    raise RuntimeError(' [!] Mean-Var stats does not match the given feature dimensions.')
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, -self.max_norm, self.max_norm)  #pylint: disable=invalid-unary-operand-type
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm + self.ref_level_db
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db /
                            self.max_norm) + self.min_level_db
                return S_denorm + self.ref_level_db
        else:
            return S_denorm

    ### Mean-STD scaling ###
    def load_stats(self, stats_path):
        stats = np.load(stats_path, allow_pickle=True).item()  #pylint: disable=unexpected-keyword-arg
        mel_mean = stats['mel_mean']
        mel_std = stats['mel_std']
        linear_mean = stats['linear_mean']
        linear_std = stats['linear_std']
        stats_config = stats['audio_config']
        # check all audio parameters used for computing stats
        skip_parameters = ['griffin_lim_iters', 'stats_path', 'do_trim_silence', 'ref_level_db', 'power']
        for key in stats_config.keys():
            if key in skip_parameters:
                continue
            if key not in ['sample_rate', 'trim_db']:
                assert stats_config[key] == self.__dict__[key],\
                    f" [!] Audio param {key} does not match the value used for computing mean-var stats. {stats_config[key]} vs {self.__dict__[key]}"
        return mel_mean, mel_std, linear_mean, linear_std, stats_config

    # pylint: disable=attribute-defined-outside-init
    def setup_scaler(self, mel_mean, mel_std, linear_mean, linear_std):
        self.mel_scaler = StandardScaler()
        self.mel_scaler.set_stats(mel_mean, mel_std)
        self.linear_scaler = StandardScaler()
        self.linear_scaler.set_stats(linear_mean, linear_std)

    ### DB and AMP conversion ###
    # pylint: disable=no-self-use
    def _amp_to_db(self, x):
        return self.spec_gain * np.log10(np.maximum(1e-5, x))

    # pylint: disable=no-self-use
    def _db_to_amp(self, x):
        return np.power(10.0, x / self.spec_gain)

    ### Preemphasis ###
    def apply_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x):
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    ### SPECTROGRAMs ###
    def _linear_to_mel(self, spectrogram):
        return np.dot(self.mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec):
        return np.maximum(1e-10, np.dot(self.inv_mel_basis, mel_spec))

    def spectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(np.abs(D))
        return self.normalize(S)

    def melspectrogram(self, y):
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        return self.normalize(S)

    def inv_spectrogram(self, spectrogram):
        """Converts spectrogram to waveform using librosa"""
        S = self.denormalize(spectrogram)
        S = self._db_to_amp(S)
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def inv_melspectrogram(self, mel_spectrogram):
        '''Converts melspectrogram to waveform using librosa'''
        D = self.denormalize(mel_spectrogram)
        S = self._db_to_amp(D)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec):
        S = self.denormalize(linear_spec)
        S = self._db_to_amp(S)
        S = self._linear_to_mel(np.abs(S))
        S = self._amp_to_db(S)
        mel = self.normalize(S)
        return mel

    ### STFT and ISTFT ###
    def _stft(self, y):
        return librosa.stft(
            y=y,
            n_fft=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )

    def _istft(self, y):
        return librosa.istft(
            y, hop_length=self.hop_length, win_length=self.win_length)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def compute_stft_paddings(self, x, pad_sides=1):
        '''compute right padding (final frame) or both sides padding (first and final frames)
        '''
        assert pad_sides in (1, 2)
        pad = (x.shape[0] // self.hop_length + 1) * self.hop_length - x.shape[0]
        if pad_sides == 1:
            return 0, pad
        return pad // 2, pad // 2 + pad % 2

    ### Compute F0 ###
    # TODO: pw causes some dep issues
    # def compute_f0(self, x):
    #     f0, t = pw.dio(
    #         x.astype(np.double),
    #         fs=self.sample_rate,
    #         f0_ceil=self.mel_fmax,
    #         frame_period=1000 * self.hop_length / self.sample_rate,
    #     )
    #     f0 = pw.stonemask(x.astype(np.double), f0, t, self.sample_rate)
    #     return f0

    ### Audio Processing ###
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
    def sound_norm(x):
        return x / abs(x).max() * 0.9

    ### save and load ###
    def load_wav(self, filename, sr=None):
        if self.resample:
            x, sr = librosa.load(filename, sr=self.sample_rate)
        elif sr is None:
            x, sr = sf.read(filename)
            assert self.sample_rate == sr, "%s vs %s"%(self.sample_rate, sr)
        else:
            x, sr = librosa.load(filename, sr=sr)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(f' [!] File cannot be trimmed for silence - {filename}')
        if self.do_sound_norm:
            x = self.sound_norm(x)
        return x

    def save_wav(self, wav, path, sample_rate=None):
        sample_rate = self.sample_rate if sample_rate is None else sample_rate
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))

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


    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -2**15, 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x, bits):
        return (x + 1.) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        return 2 * x / (2**bits - 1) - 1
