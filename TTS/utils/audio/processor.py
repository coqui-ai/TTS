from io import BytesIO
from typing import Dict, Tuple

import librosa
import numpy as np
import scipy.io.wavfile
import scipy.signal

from TTS.tts.utils.helpers import StandardScaler
from TTS.utils.audio.numpy_transforms import (
    amp_to_db,
    build_mel_basis,
    compute_f0,
    db_to_amp,
    deemphasis,
    find_endpoint,
    griffin_lim,
    load_wav,
    mel_to_spec,
    millisec_to_length,
    preemphasis,
    rms_volume_norm,
    spec_to_mel,
    stft,
    trim_silence,
    volume_norm,
)

# pylint: disable=too-many-public-methods


class AudioProcessor(object):
    """Audio Processor for TTS.

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
            self.win_length, self.hop_length = millisec_to_length(
                frame_length_ms=self.frame_length_ms, frame_shift_ms=self.frame_shift_ms, sample_rate=self.sample_rate
            )
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
        self.mel_basis = build_mel_basis(
            sample_rate=self.sample_rate,
            fft_size=self.fft_size,
            num_mels=self.num_mels,
            mel_fmax=self.mel_fmax,
            mel_fmin=self.mel_fmin,
        )
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

    ### normalization ###
    def normalize(self, S: np.ndarray) -> np.ndarray:
        """Normalize values into `[0, self.max_norm]` or `[-self.max_norm, self.max_norm]`

        Args:
            S (np.ndarray): Spectrogram to normalize.

        Raises:
            RuntimeError: Mean and variance is computed from incompatible parameters.

        Returns:
            np.ndarray: Normalized spectrogram.
        """
        # pylint: disable=no-else-return
        S = S.copy()
        if self.signal_norm:
            # mean-var scaling
            if hasattr(self, "mel_scaler"):
                if S.shape[0] == self.num_mels:
                    return self.mel_scaler.transform(S.T).T
                elif S.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.transform(S.T).T
                else:
                    raise RuntimeError(" [!] Mean-Var stats does not match the given feature dimensions.")
            # range normalization
            S -= self.ref_level_db  # discard certain range of DB assuming it is air noise
            S_norm = (S - self.min_level_db) / (-self.min_level_db)
            if self.symmetric_norm:
                S_norm = ((2 * self.max_norm) * S_norm) - self.max_norm
                if self.clip_norm:
                    S_norm = np.clip(
                        S_norm, -self.max_norm, self.max_norm  # pylint: disable=invalid-unary-operand-type
                    )
                return S_norm
            else:
                S_norm = self.max_norm * S_norm
                if self.clip_norm:
                    S_norm = np.clip(S_norm, 0, self.max_norm)
                return S_norm
        else:
            return S

    def denormalize(self, S: np.ndarray) -> np.ndarray:
        """Denormalize spectrogram values.

        Args:
            S (np.ndarray): Spectrogram to denormalize.

        Raises:
            RuntimeError: Mean and variance are incompatible.

        Returns:
            np.ndarray: Denormalized spectrogram.
        """
        # pylint: disable=no-else-return
        S_denorm = S.copy()
        if self.signal_norm:
            # mean-var scaling
            if hasattr(self, "mel_scaler"):
                if S_denorm.shape[0] == self.num_mels:
                    return self.mel_scaler.inverse_transform(S_denorm.T).T
                elif S_denorm.shape[0] == self.fft_size / 2:
                    return self.linear_scaler.inverse_transform(S_denorm.T).T
                else:
                    raise RuntimeError(" [!] Mean-Var stats does not match the given feature dimensions.")
            if self.symmetric_norm:
                if self.clip_norm:
                    S_denorm = np.clip(
                        S_denorm, -self.max_norm, self.max_norm  # pylint: disable=invalid-unary-operand-type
                    )
                S_denorm = ((S_denorm + self.max_norm) * -self.min_level_db / (2 * self.max_norm)) + self.min_level_db
                return S_denorm + self.ref_level_db
            else:
                if self.clip_norm:
                    S_denorm = np.clip(S_denorm, 0, self.max_norm)
                S_denorm = (S_denorm * -self.min_level_db / self.max_norm) + self.min_level_db
                return S_denorm + self.ref_level_db
        else:
            return S_denorm

    ### Mean-STD scaling ###
    def load_stats(self, stats_path: str) -> Tuple[np.array, np.array, np.array, np.array, Dict]:
        """Loading mean and variance statistics from a `npy` file.

        Args:
            stats_path (str): Path to the `npy` file containing

        Returns:
            Tuple[np.array, np.array, np.array, np.array, Dict]: loaded statistics and the config used to
                compute them.
        """
        stats = np.load(stats_path, allow_pickle=True).item()  # pylint: disable=unexpected-keyword-arg
        mel_mean = stats["mel_mean"]
        mel_std = stats["mel_std"]
        linear_mean = stats["linear_mean"]
        linear_std = stats["linear_std"]
        stats_config = stats["audio_config"]
        # check all audio parameters used for computing stats
        skip_parameters = ["griffin_lim_iters", "stats_path", "do_trim_silence", "ref_level_db", "power"]
        for key in stats_config.keys():
            if key in skip_parameters:
                continue
            if key not in ["sample_rate", "trim_db"]:
                assert (
                    stats_config[key] == self.__dict__[key]
                ), f" [!] Audio param {key} does not match the value used for computing mean-var stats. {stats_config[key]} vs {self.__dict__[key]}"
        return mel_mean, mel_std, linear_mean, linear_std, stats_config

    # pylint: disable=attribute-defined-outside-init
    def setup_scaler(
        self, mel_mean: np.ndarray, mel_std: np.ndarray, linear_mean: np.ndarray, linear_std: np.ndarray
    ) -> None:
        """Initialize scaler objects used in mean-std normalization.

        Args:
            mel_mean (np.ndarray): Mean for melspectrograms.
            mel_std (np.ndarray): STD for melspectrograms.
            linear_mean (np.ndarray): Mean for full scale spectrograms.
            linear_std (np.ndarray): STD for full scale spectrograms.
        """
        self.mel_scaler = StandardScaler()
        self.mel_scaler.set_stats(mel_mean, mel_std)
        self.linear_scaler = StandardScaler()
        self.linear_scaler.set_stats(linear_mean, linear_std)

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
        return preemphasis(x=x, coef=self.preemphasis)

    def apply_inv_preemphasis(self, x: np.ndarray) -> np.ndarray:
        """Reverse pre-emphasis."""
        return deemphasis(x=x, coef=self.preemphasis)

    ### SPECTROGRAMs ###
    def spectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a spectrogram from a waveform.

        Args:
            y (np.ndarray): Waveform.

        Returns:
            np.ndarray: Spectrogram.
        """
        if self.preemphasis != 0:
            y = self.apply_preemphasis(y)
        D = stft(
            y=y,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )
        if self.do_amp_to_db_linear:
            S = amp_to_db(x=np.abs(D), gain=self.spec_gain, base=self.base)
        else:
            S = np.abs(D)
        return self.normalize(S).astype(np.float32)

    def melspectrogram(self, y: np.ndarray) -> np.ndarray:
        """Compute a melspectrogram from a waveform."""
        if self.preemphasis != 0:
            y = self.apply_preemphasis(y)
        D = stft(
            y=y,
            fft_size=self.fft_size,
            hop_length=self.hop_length,
            win_length=self.win_length,
            pad_mode=self.stft_pad_mode,
        )
        S = spec_to_mel(spec=np.abs(D), mel_basis=self.mel_basis)
        if self.do_amp_to_db_mel:
            S = amp_to_db(x=S, gain=self.spec_gain, base=self.base)

        return self.normalize(S).astype(np.float32)

    def inv_spectrogram(self, spectrogram: np.ndarray) -> np.ndarray:
        """Convert a spectrogram to a waveform using Griffi-Lim vocoder."""
        S = self.denormalize(spectrogram)
        S = db_to_amp(x=S, gain=self.spec_gain, base=self.base)
        # Reconstruct phase
        W = self._griffin_lim(S**self.power)
        return self.apply_inv_preemphasis(W) if self.preemphasis != 0 else W

    def inv_melspectrogram(self, mel_spectrogram: np.ndarray) -> np.ndarray:
        """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
        D = self.denormalize(mel_spectrogram)
        S = db_to_amp(x=D, gain=self.spec_gain, base=self.base)
        S = mel_to_spec(mel=S, mel_basis=self.mel_basis)  # Convert back to linear
        W = self._griffin_lim(S**self.power)
        return self.apply_inv_preemphasis(W) if self.preemphasis != 0 else W

    def out_linear_to_mel(self, linear_spec: np.ndarray) -> np.ndarray:
        """Convert a full scale linear spectrogram output of a network to a melspectrogram.

        Args:
            linear_spec (np.ndarray): Normalized full scale linear spectrogram.

        Returns:
            np.ndarray: Normalized melspectrogram.
        """
        S = self.denormalize(linear_spec)
        S = db_to_amp(x=S, gain=self.spec_gain, base=self.base)
        S = spec_to_mel(spec=np.abs(S), mel_basis=self.mel_basis)
        S = amp_to_db(x=S, gain=self.spec_gain, base=self.base)
        mel = self.normalize(S)
        return mel

    def _griffin_lim(self, S):
        return griffin_lim(
            spec=S,
            num_iter=self.griffin_lim_iters,
            hop_length=self.hop_length,
            win_length=self.win_length,
            fft_size=self.fft_size,
            pad_mode=self.stft_pad_mode,
        )

    def compute_f0(self, x: np.ndarray) -> np.ndarray:
        """Compute pitch (f0) of a waveform using the same parameters used for computing melspectrogram.

        Args:
            x (np.ndarray): Waveform.

        Returns:
            np.ndarray: Pitch.

        Examples:
            >>> WAV_FILE = filename = librosa.example('vibeace')
            >>> from TTS.config import BaseAudioConfig
            >>> from TTS.utils.audio import AudioProcessor
            >>> conf = BaseAudioConfig(pitch_fmax=640, pitch_fmin=1)
            >>> ap = AudioProcessor(**conf)
            >>> wav = ap.load_wav(WAV_FILE, sr=ap.sample_rate)[:5 * ap.sample_rate]
            >>> pitch = ap.compute_f0(wav)
        """
        # align F0 length to the spectrogram length
        if len(x) % self.hop_length == 0:
            x = np.pad(x, (0, self.hop_length // 2), mode=self.stft_pad_mode)

        f0 = compute_f0(
            x=x,
            pitch_fmax=self.pitch_fmax,
            pitch_fmin=self.pitch_fmin,
            hop_length=self.hop_length,
            win_length=self.win_length,
            sample_rate=self.sample_rate,
            stft_pad_mode=self.stft_pad_mode,
            center=True,
        )

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
        return find_endpoint(
            wav=wav,
            trim_db=self.trim_db,
            sample_rate=self.sample_rate,
            min_silence_sec=min_silence_sec,
            gain=self.spec_gain,
            base=self.base,
        )

    def trim_silence(self, wav):
        """Trim silent parts with a threshold and 0.01 sec margin"""
        return trim_silence(
            wav=wav,
            sample_rate=self.sample_rate,
            trim_db=self.trim_db,
            win_length=self.win_length,
            hop_length=self.hop_length,
        )

    @staticmethod
    def sound_norm(x: np.ndarray) -> np.ndarray:
        """Normalize the volume of an audio signal.

        Args:
            x (np.ndarray): Raw waveform.

        Returns:
            np.ndarray: Volume normalized waveform.
        """
        return volume_norm(x=x)

    def rms_volume_norm(self, x: np.ndarray, db_level: float = None) -> np.ndarray:
        """Normalize the volume based on RMS of the signal.

        Args:
            x (np.ndarray): Raw waveform.

        Returns:
            np.ndarray: RMS normalized waveform.
        """
        if db_level is None:
            db_level = self.db_level
        return rms_volume_norm(x=x, db_level=db_level)

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
        if sr is not None:
            x = load_wav(filename=filename, sample_rate=sr, resample=True)
        else:
            x = load_wav(filename=filename, sample_rate=self.sample_rate, resample=self.resample)
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

    def save_wav(self, wav: np.ndarray, path: str, sr: int = None, pipe_out=None) -> None:
        """Save a waveform to a file using Scipy.

        Args:
            wav (np.ndarray): Waveform to save.
            path (str): Path to a output file.
            sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
        """
        if self.do_rms_norm:
            wav_norm = self.rms_volume_norm(wav, self.db_level) * 32767
        else:
            wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))

        wav_norm = wav_norm.astype(np.int16)
        if pipe_out:
            wav_buffer = BytesIO()
            scipy.io.wavfile.write(wav_buffer, sr if sr else self.sample_rate, wav_norm)
            wav_buffer.seek(0)
            pipe_out.buffer.write(wav_buffer.read())
        scipy.io.wavfile.write(path, sr if sr else self.sample_rate, wav_norm)

    def get_duration(self, filename: str) -> float:
        """Get the duration of a wav file using Librosa.

        Args:
            filename (str): Path to the wav file.
        """
        return librosa.get_duration(filename=filename)
