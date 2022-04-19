import librosa
import numpy as np
import pyworld as pw
import scipy.io.wavfile
import scipy.signal
import soundfile as sf


# pylint: disable=too-many-public-methods
class AudioProcessor(object):
    """Audio Processor for TTS used by all the data pipelines.

    TODO: Make this a dataclass to replace `BaseAudioConfig`.

    Note:
        All the class arguments are set to default values to enable a flexible initialization
        of the class with the model config. They are not meaningful for all the arguments.

    Args:
        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        resample (bool, optional):
            enable/disable resampling of the audio clips when the target sampling rate does not match the original sampling rate. Defaults to False.

        num_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        log_func (int, optional):
            log exponent used for converting spectrogram aplitude to DB.

        min_level_db (int, optional):
            minimum db threshold for the computed melspectrograms. Defaults to None.

        frame_shift_ms (int, optional):
            milliseconds of frames between STFT columns. Defaults to None.

        frame_length_ms (int, optional):
            milliseconds of STFT window length. Defaults to None.

        hop_length (int, optional):
            number of frames between STFT columns. Used if ```frame_shift_ms``` is None. Defaults to None.

        win_length (int, optional):
            STFT window length. Used if ```frame_length_ms``` is None. Defaults to None.

        ref_level_db (int, optional):
            reference DB level to avoid background noise. In general <20DB corresponds to the air noise. Defaults to None.

        fft_size (int, optional):
            FFT window size for STFT. Defaults to 1024.

        power (int, optional):
            Exponent value applied to the spectrogram before GriffinLim. Defaults to None.

        preemphasis (float, optional):
            Preemphasis coefficient. Preemphasis is disabled if == 0.0. Defaults to 0.0.

        signal_norm (bool, optional):
            enable/disable signal normalization. Defaults to None.

        symmetric_norm (bool, optional):
            enable/disable symmetric normalization. If set True normalization is performed in the range [-k, k] else [0, k], Defaults to None.

        max_norm (float, optional):
            ```k``` defining the normalization range. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        pitch_fmin (int, optional):
            minimum filter frequency for computing pitch. Defaults to None.

        pitch_fmax (int, optional):
            maximum filter frequency for computing pitch. Defaults to None.

        spec_gain (int, optional):
            gain applied when converting amplitude to DB. Defaults to 20.

        stft_pad_mode (str, optional):
            Padding mode for STFT. Defaults to 'reflect'.

        clip_norm (bool, optional):
            enable/disable clipping the our of range values in the normalized audio signal. Defaults to True.

        griffin_lim_iters (int, optional):
            Number of GriffinLim iterations. Defaults to None.

        do_trim_silence (bool, optional):
            enable/disable silence trimming when loading the audio signal. Defaults to False.

        trim_db (int, optional):
            DB threshold used for silence trimming. Defaults to 60.

        do_sound_norm (bool, optional):
            enable/disable signal normalization. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to True.

        do_amp_to_db_mel (bool, optional):
            enable/disable amplitude to dB conversion of mel spectrograms. Defaults to True.

        do_rms_norm (bool, optional):
            enable/disable RMS volume normalization when loading an audio file. Defaults to False.

        db_level (int, optional):
            dB level used for rms normalization. The range is -99 to 0. Defaults to None.

        stats_path (str, optional):
            Path to the computed stats file. Defaults to None.

        verbose (bool, optional):
            enable/disable logging. Defaults to True.

    """

    def __init__(
        self,
        sample_rate=None,
        resample=False,
        num_mels=None,
        log_func="np.log10",
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
        pitch_fmax=None,
        pitch_fmin=None,
        spec_gain=20,
        stft_pad_mode="reflect",
        clip_norm=True,
        griffin_lim_iters=None,
        do_trim_silence=False,
        trim_db=60,
        do_sound_norm=False,
        do_amp_to_db_linear=True,
        do_amp_to_db_mel=True,
        do_rms_norm=False,
        db_level=None,
        stats_path=None,
        verbose=True,
        **_,
    ):

        # setup class attributed
        self.sample_rate = sample_rate
        self.resample = resample
        self.num_mels = num_mels
        self.log_func = log_func
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
        self.pitch_fmin = pitch_fmin
        self.pitch_fmax = pitch_fmax
        self.spec_gain = float(spec_gain)
        self.stft_pad_mode = stft_pad_mode
        self.max_norm = 1.0 if max_norm is None else float(max_norm)
        self.clip_norm = clip_norm
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.do_sound_norm = do_sound_norm
        self.do_amp_to_db_linear = do_amp_to_db_linear
        self.do_amp_to_db_mel = do_amp_to_db_mel
        self.do_rms_norm = do_rms_norm
        self.db_level = db_level
        self.stats_path = stats_path
        # setup exp_func for db to amp conversion
        if log_func == "np.log":
            self.base = np.e
        elif log_func == "np.log10":
            self.base = 10
        else:
            raise ValueError(" [!] unknown `log_func` value.")
        # setup stft parameters
        if hop_length is None:
            # compute stft parameters from given time values
            self.hop_length, self.win_length = self._stft_parameters()
        else:
            # use stft parameters from config file
            self.hop_length = hop_length
            self.win_length = win_length
        assert min_level_db != 0.0, " [!] min_level_db is 0"
        assert (
            self.win_length <= self.fft_size
        ), f" [!] win_length cannot be larger than fft_size - {self.win_length} vs {self.fft_size}"
        members = vars(self)
        if verbose:
            print(" > Setting up Audio Processor...")
            for key, value in members.items():
                print(" | > {}:{}".format(key, value))
        # create spectrogram utils
        self.mel_basis = self._build_mel_basis()
        self.inv_mel_basis = np.linalg.pinv(self._build_mel_basis())
        # setup scaler
        if stats_path and signal_norm:
            mel_mean, mel_std, linear_mean, linear_std, _ = self.load_stats(stats_path)
            self.setup_scaler(mel_mean, mel_std, linear_mean, linear_std)
            self.signal_norm = True
            self.max_norm = None
            self.clip_norm = None
            self.symmetric_norm = None

    @staticmethod
    def init_from_config(config: "Coqpit", verbose=True):
        if "audio" in config:
            return AudioProcessor(verbose=verbose, **config.audio)
        return AudioProcessor(verbose=verbose, **config)

    ### setting up the parameters ###

    ### DB and AMP conversion ###
    # pylint: disable=no-self-use
    def _amp_to_db(self, x: np.ndarray) -> np.ndarray:
        """Convert amplitude values to decibels.

        Args:
            x (np.ndarray): Amplitude spectrogram.

        Returns:
            np.ndarray: Decibels spectrogram.
        """
        return self.spec_gain * _log(np.maximum(1e-5, x), self.base)

    # pylint: disable=no-self-use
    def _db_to_amp(self, x: np.ndarray) -> np.ndarray:
        """Convert decibels spectrogram to amplitude spectrogram.

        Args:
            x (np.ndarray): Decibels spectrogram.

        Returns:
            np.ndarray: Amplitude spectrogram.
        """
        return _exp(x / self.spec_gain, self.base)

    ### Preemphasis ###
    def apply_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis to the audio signal. Useful to reduce the correlation between neighbouring signal values.

        Args:
            x (np.ndarray): Audio signal.

        Raises:
            RuntimeError: Preemphasis coeff is set to 0.

        Returns:
            np.ndarray: Decorrelated audio signal.
        """
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1, -self.preemphasis], [1], x)

    def apply_inv_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Reverse pre-emphasis."""
        if self.preemphasis == 0:
            raise RuntimeError(" [!] Preemphasis is set 0.0.")
        return scipy.signal.lfilter([1], [1, -self.preemphasis], x)

    ### SPECTROGRAMs ###
    def _linear_to_mel(self, spectrogram: np.ndarray) -> np.ndarray:
        """Project a full scale spectrogram to a melspectrogram.

        Args:
            spectrogram (np.ndarray): Full scale spectrogram.

        Returns:
            np.ndarray: Melspectrogram
        """
        return np.dot(self.mel_basis, spectrogram)

    def _mel_to_linear(self, mel_spec: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to full scale spectrogram."""
        return np.maximum(1e-10, np.dot(self.inv_mel_basis, mel_spec))

    def spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a spectrogram from a waveform.

        Args:
            y (np.ndarray): Waveform.

        Returns:
            np.ndarray: Spectrogram.
        """
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        if self.do_amp_to_db_linear:
            S = self._amp_to_db(np.abs(D))
        else:
            S = np.abs(D)
        return self.normalize(S).astype(np.float32)

    def melspectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a melspectrogram from a waveform."""
        if self.preemphasis != 0:
            D = self._stft(self.apply_preemphasis(y))
        else:
            D = self._stft(y)
        if self.do_amp_to_db_mel:
            S = self._amp_to_db(self._linear_to_mel(np.abs(D)))
        else:
            S = self._linear_to_mel(np.abs(D))
        return self.normalize(S).astype(np.float32)

    def inv_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert a spectrogram to a waveform using Griffi-Lim vocoder."""
        S = self.denormalize(spectrogram)
        S = self._db_to_amp(S)
        # Reconstruct phase
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def inv_melspectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
        D = self.denormalize(mel_spectrogram)
        S = self._db_to_amp(D)
        S = self._mel_to_linear(S)  # Convert back to linear
        if self.preemphasis != 0:
            return self.apply_inv_preemphasis(self._griffin_lim(S**self.power))
        return self._griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec: np.ndarray) -> np.ndarray:
        """Convert a full scale linear spectrogram output of a network to a melspectrogram.

        Args:
            linear_spec (np.ndarray): Normalized full scale linear spectrogram.

        Returns:
            np.ndarray: Normalized melspectrogram.
        """
        S = self.denormalize(linear_spec)
        S = self._db_to_amp(S)
        S = self._linear_to_mel(np.abs(S))
        S = self._amp_to_db(S)
        mel = self.normalize(S)
        return mel

    ### STFT and ISTFT ###
    def _stft(self, y: np.ndarray) -> np.ndarray:
        """Librosa STFT wrapper.

        Args:
            y (np.ndarray): Audio signal.

        Returns:
            np.ndarray: Complex number array.
        """
        return librosa.stft(
            y=y,
            n_fft=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
            window="hann",
            center=True,
        )

    def _istft(self, y: np.ndarray) -> np.ndarray:
        """Librosa iSTFT wrapper."""
        return librosa.istft(y, hop_length=self.hop_length, win_length=self.win_length)

    def _griffin_lim(self, S):
        angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
        S_complex = np.abs(S).astype(np.complex)
        y = self._istft(S_complex * angles)
        if not np.isfinite(y).all():
            print(" [!] Waveform is not finite everywhere. Skipping the GL.")
            return np.array([0.0])
        for _ in range(self.griffin_lim_iters):
            angles = np.exp(1j * np.angle(self._stft(y)))
            y = self._istft(S_complex * angles)
        return y

    def compute_stft_paddings(self, x, pad_sides=1):
        """Compute paddings used by Librosa's STFT. Compute right padding (final frame) or both sides padding
        (first and final frames)"""
        assert pad_sides in (1, 2)
        pad = (x.shape[0] // self.hop_length + 1) * self.hop_length - x.shape[0]
        if pad_sides == 1:
            return 0, pad
        return pad // 2, pad // 2 + pad % 2

    def compute_f0(self, x: np.ndarray) -> np.ndarray:
        """Compute pitch (f0) of a waveform using the same parameters used for computing melspectrogram.

        Args:
            x (np.ndarray): Waveform.

        Returns:
            np.ndarray: Pitch.

        Examples:
            >>> WAV_FILE = filename = librosa.util.example_audio_file()
            >>> from TTS.config import BaseAudioConfig
            >>> from TTS.utils.audio.processor import AudioProcessor            >>> conf = BaseAudioConfig(pitch_fmax=8000)
            >>> ap = AudioProcessor(**conf)
            >>> wav = ap.load_wav(WAV_FILE, sr=22050)[:5 * 22050]
            >>> pitch = ap.compute_f0(wav)
        """
        assert self.pitch_fmax is not None, " [!] Set `pitch_fmax` before caling `compute_f0`."
        # align F0 length to the spectrogram length
        if len(x) % self.hop_length == 0:
            x = np.pad(x, (0, self.hop_length // 2), mode="reflect")

        f0, t = pw.dio(
            x.astype(np.double),
            fs=self.sample_rate,
            f0_ceil=self.pitch_fmax,
            frame_period=1000 * self.hop_length / self.sample_rate,
        )
        f0 = pw.stonemask(x.astype(np.double), f0, t, self.sample_rate)
        return f0

    ### Audio Processing ###
    def find_endpoint(self, wav: np.ndarray, min_silence_sec=0.8) -> int:
        """Find the last point without silence at the end of a audio signal.

        Args:
            wav (np.ndarray): Audio signal.
            threshold_db (int, optional): Silence threshold in decibels. Defaults to -40.
            min_silence_sec (float, optional): Ignore silences that are shorter then this in secs. Defaults to 0.8.

        Returns:
            int: Last point without silence.
        """
        window_length = int(self.sample_rate * min_silence_sec)
        hop_length = int(window_length / 4)
        threshold = self._db_to_amp(-self.trim_db)
        for x in range(hop_length, len(wav) - window_length, hop_length):
            if np.max(wav[x : x + window_length]) < threshold:
                return x + hop_length
        return len(wav)

    def trim_silence(self, wav):
        """Trim silent parts with a threshold and 0.01 sec margin"""
        margin = int(self.sample_rate * 0.01)
        wav = wav[margin:-margin]
        return librosa.effects.trim(wav, top_db=self.trim_db, frame_length=self.win_length, hop_length=self.hop_length)[
            0
        ]

    @staticmethod
    def sound_norm(x: np.ndarray) -> np.ndarray:
        """Normalize the volume of an audio signal.

        Args:
            x (np.ndarray): Raw waveform.

        Returns:
            np.ndarray: Volume normalized waveform.
        """
        return x / abs(x).max() * 0.95

    @staticmethod
    def _rms_norm(wav, db_level=-27):
        r = 10 ** (db_level / 20)
        a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
        return wav * a

    def rms_volume_norm(self, x: np.ndarray, db_level: float = None) -> np.ndarray:
        """Normalize the volume based on RMS of the signal.

        Args:
            x (np.ndarray): Raw waveform.

        Returns:
            np.ndarray: RMS normalized waveform.
        """
        if db_level is None:
            db_level = self.db_level
        assert -99 <= db_level <= 0, " [!] db_level should be between -99 and 0"
        wav = self._rms_norm(x, db_level)
        return wav

    ### save and load ###
    def load_wav(self, filename: str, sr: int = None) -> np.ndarray:
        """Read a wav file using Librosa and optionally resample, silence trim, volume normalize.

        Resampling slows down loading the file significantly. Therefore it is recommended to resample the file before.

        Args:
            filename (str): Path to the wav file.
            sr (int, optional): Sampling rate for resampling. Defaults to None.

        Returns:
            np.ndarray: Loaded waveform.
        """
        if self.resample:
            # loading with resampling. It is significantly slower.
            x, sr = librosa.load(filename, sr=self.sample_rate)
        elif sr is None:
            # SF is faster than librosa for loading files
            x, sr = sf.read(filename)
            assert self.sample_rate == sr, "%s vs %s" % (self.sample_rate, sr)
        else:
            x, sr = librosa.load(filename, sr=sr)
        if self.do_trim_silence:
            try:
                x = self.trim_silence(x)
            except ValueError:
                print(f" [!] File cannot be trimmed for silence - {filename}")
        if self.do_sound_norm:
            x = self.sound_norm(x)
        if self.do_rms_norm:
            x = self.rms_volume_norm(x, self.db_level)
        return x

    def save_wav(self, wav: np.ndarray, path: str, sr: int = None) -> None:
        """Save a waveform to a file using Scipy.

        Args:
            wav (np.ndarray): Waveform to save.
            path (str): Path to a output file.
            sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
        """
        if self.do_rms_norm:
            wav_norm = self.rms_volume_norm(wav, self.db_level) * 32767
        else:
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

        scipy.io.wavfile.write(path, sr if sr else self.sample_rate, wav_norm.astype(np.int16))

    def get_duration(self, filename: str) -> float:
        """Get the duration of a wav file using Librosa.

        Args:
            filename (str): Path to the wav file.
        """
        return librosa.get_duration(filename)

    @staticmethod
    def mulaw_encode(wav: np.ndarray, qc: int) -> np.ndarray:
        mu = 2**qc - 1
        # wav_abs = np.minimum(np.abs(wav), 1.0)
        signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1.0 + mu)
        # Quantize signal to the specified number of levels.
        signal = (signal + 1) / 2 * mu + 0.5
        return np.floor(
            signal,
        )

    @staticmethod
    def mulaw_decode(wav, qc):
        """Recovers waveform from quantized values."""
        mu = 2**qc - 1
        x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
        return x

    @staticmethod
    def encode_16bits(x):
        return np.clip(x * 2**15, -(2**15), 2**15 - 1).astype(np.int16)

    @staticmethod
    def quantize(x: np.ndarray, bits: int) -> np.ndarray:
        """Quantize a waveform to a given number of bits.

        Args:
            x (np.ndarray): Waveform to quantize. Must be normalized into the range `[-1, 1]`.
            bits (int): Number of quantization bits.

        Returns:
            np.ndarray: Quantized waveform.
        """
        return (x + 1.0) * (2**bits - 1) / 2

    @staticmethod
    def dequantize(x, bits):
        """Dequantize a waveform from the given number of bits."""
        return 2 * x / (2**bits - 1) - 1
