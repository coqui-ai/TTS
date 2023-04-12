from typing import Tuple

import librosa
import numpy as np
import scipy
import soundfile as sf
from librosa import magphase, pyin

# For using kwargs
# pylint: disable=unused-argument


def build_mel_basis(
    *,
    sample_rate: int = None,
    fft_size: int = None,
    num_mels: int = None,
    mel_fmax: int = None,
    mel_fmin: int = None,
    **kwargs,
) -> np.ndarray:
    """Build melspectrogram basis.

    Returns:
        np.ndarray: melspectrogram basis.
    """
    if mel_fmax is not None:
        assert mel_fmax <= sample_rate // 2
        assert mel_fmax - mel_fmin > 0
    return librosa.filters.mel(sr=sample_rate, n_fft=fft_size, n_mels=num_mels, fmin=mel_fmin, fmax=mel_fmax)


def millisec_to_length(
    *, frame_length_ms: int = None, frame_shift_ms: int = None, sample_rate: int = None, **kwargs
) -> Tuple[int, int]:
    """Compute hop and window length from milliseconds.

    Returns:
        Tuple[int, int]: hop length and window length for STFT.
    """
    factor = frame_length_ms / frame_shift_ms
    assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
    win_length = int(frame_length_ms / 1000.0 * sample_rate)
    hop_length = int(win_length / float(factor))
    return win_length, hop_length


def _log(x, base):
    if base == 10:
        return np.log10(x)
    return np.log(x)


def _exp(x, base):
    if base == 10:
        return np.power(10, x)
    return np.exp(x)


def amp_to_db(*, x: np.ndarray = None, gain: float = 1, base: int = 10, **kwargs) -> np.ndarray:
    """Convert amplitude values to decibels.

    Args:
        x (np.ndarray): Amplitude spectrogram.
        gain (float): Gain factor. Defaults to 1.
        base (int): Logarithm base. Defaults to 10.

    Returns:
        np.ndarray: Decibels spectrogram.
    """
    assert (x < 0).sum() == 0, " [!] Input values must be non-negative."
    return gain * _log(np.maximum(1e-8, x), base)


# pylint: disable=no-self-use
def db_to_amp(*, x: np.ndarray = None, gain: float = 1, base: int = 10, **kwargs) -> np.ndarray:
    """Convert decibels spectrogram to amplitude spectrogram.

    Args:
        x (np.ndarray): Decibels spectrogram.
        gain (float): Gain factor. Defaults to 1.
        base (int): Logarithm base. Defaults to 10.

    Returns:
        np.ndarray: Amplitude spectrogram.
    """
    return _exp(x / gain, base)


def preemphasis(*, x: np.ndarray, coef: float = 0.97, **kwargs) -> np.ndarray:
    """Apply pre-emphasis to the audio signal. Useful to reduce the correlation between neighbouring signal values.

    Args:
        x (np.ndarray): Audio signal.

    Raises:
        RuntimeError: Preemphasis coeff is set to 0.

    Returns:
        np.ndarray: Decorrelated audio signal.
    """
    if coef == 0:
        raise RuntimeError(" [!] Preemphasis is set 0.0.")
    return scipy.signal.lfilter([1, -coef], [1], x)


def deemphasis(*, x: np.ndarray = None, coef: float = 0.97, **kwargs) -> np.ndarray:
    """Reverse pre-emphasis."""
    if coef == 0:
        raise RuntimeError(" [!] Preemphasis is set 0.0.")
    return scipy.signal.lfilter([1], [1, -coef], x)


def spec_to_mel(*, spec: np.ndarray, mel_basis: np.ndarray = None, **kwargs) -> np.ndarray:
    """Convert a full scale linear spectrogram output of a network to a melspectrogram.

    Args:
        spec (np.ndarray): Normalized full scale linear spectrogram.

    Shapes:
        - spec: :math:`[C, T]`

    Returns:
        np.ndarray: Normalized melspectrogram.
    """
    return np.dot(mel_basis, spec)


def mel_to_spec(*, mel: np.ndarray = None, mel_basis: np.ndarray = None, **kwargs) -> np.ndarray:
    """Convert a melspectrogram to full scale spectrogram."""
    assert (mel < 0).sum() == 0, " [!] Input values must be non-negative."
    inv_mel_basis = np.linalg.pinv(mel_basis)
    return np.maximum(1e-10, np.dot(inv_mel_basis, mel))


def wav_to_spec(*, wav: np.ndarray = None, **kwargs) -> np.ndarray:
    """Compute a spectrogram from a waveform.

    Args:
        wav (np.ndarray): Waveform. Shape :math:`[T_wav,]`

    Returns:
        np.ndarray: Spectrogram. Shape :math:`[C, T_spec]`. :math:`T_spec == T_wav / hop_length`
    """
    D = stft(y=wav, **kwargs)
    S = np.abs(D)
    return S.astype(np.float32)


def wav_to_mel(*, wav: np.ndarray = None, mel_basis=None, **kwargs) -> np.ndarray:
    """Compute a melspectrogram from a waveform."""
    D = stft(y=wav, **kwargs)
    S = spec_to_mel(spec=np.abs(D), mel_basis=mel_basis, **kwargs)
    return S.astype(np.float32)


def spec_to_wav(*, spec: np.ndarray, power: float = 1.5, **kwargs) -> np.ndarray:
    """Convert a spectrogram to a waveform using Griffi-Lim vocoder."""
    S = spec.copy()
    return griffin_lim(spec=S**power, **kwargs)


def mel_to_wav(*, mel: np.ndarray = None, power: float = 1.5, **kwargs) -> np.ndarray:
    """Convert a melspectrogram to a waveform using Griffi-Lim vocoder."""
    S = mel.copy()
    S = mel_to_spec(mel=S, mel_basis=kwargs["mel_basis"])  # Convert back to linear
    return griffin_lim(spec=S**power, **kwargs)


### STFT and ISTFT ###
def stft(
    *,
    y: np.ndarray = None,
    fft_size: int = None,
    hop_length: int = None,
    win_length: int = None,
    pad_mode: str = "reflect",
    window: str = "hann",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    """Librosa STFT wrapper.

    Check http://librosa.org/doc/main/generated/librosa.stft.html argument details.

    Returns:
        np.ndarray: Complex number array.
    """
    return librosa.stft(
        y=y,
        n_fft=fft_size,
        hop_length=hop_length,
        win_length=win_length,
        pad_mode=pad_mode,
        window=window,
        center=center,
    )


def istft(
    *,
    y: np.ndarray = None,
    fft_size: int = None,
    hop_length: int = None,
    win_length: int = None,
    window: str = "hann",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    """Librosa iSTFT wrapper.

    Check http://librosa.org/doc/main/generated/librosa.istft.html argument details.

    Returns:
        np.ndarray: Complex number array.
    """
    return librosa.istft(y, hop_length=hop_length, win_length=win_length, center=center, window=window)


def griffin_lim(*, spec: np.ndarray = None, num_iter=60, **kwargs) -> np.ndarray:
    angles = np.exp(2j * np.pi * np.random.rand(*spec.shape))
    S_complex = np.abs(spec).astype(np.complex)
    y = istft(y=S_complex * angles, **kwargs)
    if not np.isfinite(y).all():
        print(" [!] Waveform is not finite everywhere. Skipping the GL.")
        return np.array([0.0])
    for _ in range(num_iter):
        angles = np.exp(1j * np.angle(stft(y=y, **kwargs)))
        y = istft(y=S_complex * angles, **kwargs)
    return y


def compute_stft_paddings(
    *, x: np.ndarray = None, hop_length: int = None, pad_two_sides: bool = False, **kwargs
) -> Tuple[int, int]:
    """Compute paddings used by Librosa's STFT. Compute right padding (final frame) or both sides padding
    (first and final frames)"""
    pad = (x.shape[0] // hop_length + 1) * hop_length - x.shape[0]
    if not pad_two_sides:
        return 0, pad
    return pad // 2, pad // 2 + pad % 2


def compute_f0(
    *,
    x: np.ndarray = None,
    pitch_fmax: float = None,
    pitch_fmin: float = None,
    hop_length: int = None,
    win_length: int = None,
    sample_rate: int = None,
    stft_pad_mode: str = "reflect",
    center: bool = True,
    **kwargs,
) -> np.ndarray:
    """Compute pitch (f0) of a waveform using the same parameters used for computing melspectrogram.

    Args:
        x (np.ndarray): Waveform. Shape :math:`[T_wav,]`
        pitch_fmax (float): Pitch max value.
        pitch_fmin (float): Pitch min value.
        hop_length (int): Number of frames between STFT columns.
        win_length (int): STFT window length.
        sample_rate (int): Audio sampling rate.
        stft_pad_mode (str): Padding mode for STFT.
        center (bool): Centered padding.

    Returns:
        np.ndarray: Pitch. Shape :math:`[T_pitch,]`. :math:`T_pitch == T_wav / hop_length`

    Examples:
        >>> WAV_FILE = filename = librosa.example('vibeace')
        >>> from TTS.config import BaseAudioConfig
        >>> from TTS.utils.audio import AudioProcessor
        >>> conf = BaseAudioConfig(pitch_fmax=640, pitch_fmin=1)
        >>> ap = AudioProcessor(**conf)
        >>> wav = ap.load_wav(WAV_FILE, sr=ap.sample_rate)[:5 * ap.sample_rate]
        >>> pitch = ap.compute_f0(wav)
    """
    assert pitch_fmax is not None, " [!] Set `pitch_fmax` before caling `compute_f0`."
    assert pitch_fmin is not None, " [!] Set `pitch_fmin` before caling `compute_f0`."

    f0, voiced_mask, _ = pyin(
        y=x.astype(np.double),
        fmin=pitch_fmin,
        fmax=pitch_fmax,
        sr=sample_rate,
        frame_length=win_length,
        win_length=win_length // 2,
        hop_length=hop_length,
        pad_mode=stft_pad_mode,
        center=center,
        n_thresholds=100,
        beta_parameters=(2, 18),
        boltzmann_parameter=2,
        resolution=0.1,
        max_transition_rate=35.92,
        switch_prob=0.01,
        no_trough_prob=0.01,
    )
    f0[~voiced_mask] = 0.0

    return f0


def compute_energy(y: np.ndarray, **kwargs) -> np.ndarray:
    """Compute energy of a waveform using the same parameters used for computing melspectrogram.
    Args:
      x (np.ndarray): Waveform. Shape :math:`[T_wav,]`
    Returns:
      np.ndarray: energy. Shape :math:`[T_energy,]`. :math:`T_energy == T_wav / hop_length`
    Examples:
      >>> WAV_FILE = filename = librosa.example('vibeace')
      >>> from TTS.config import BaseAudioConfig
      >>> from TTS.utils.audio import AudioProcessor
      >>> conf = BaseAudioConfig()
      >>> ap = AudioProcessor(**conf)
      >>> wav = ap.load_wav(WAV_FILE, sr=ap.sample_rate)[:5 * ap.sample_rate]
      >>> energy = ap.compute_energy(wav)
    """
    x = stft(y=y, **kwargs)
    mag, _ = magphase(x)
    energy = np.sqrt(np.sum(mag**2, axis=0))
    return energy


### Audio Processing ###
def find_endpoint(
    *,
    wav: np.ndarray = None,
    trim_db: float = -40,
    sample_rate: int = None,
    min_silence_sec=0.8,
    gain: float = None,
    base: int = None,
    **kwargs,
) -> int:
    """Find the last point without silence at the end of a audio signal.

    Args:
        wav (np.ndarray): Audio signal.
        threshold_db (int, optional): Silence threshold in decibels. Defaults to -40.
        min_silence_sec (float, optional): Ignore silences that are shorter then this in secs. Defaults to 0.8.
        gian (float, optional): Gain to be used to convert trim_db to trim_amp. Defaults to None.
        base (int, optional): Base of the logarithm used to convert trim_db to trim_amp. Defaults to 10.

    Returns:
        int: Last point without silence.
    """
    window_length = int(sample_rate * min_silence_sec)
    hop_length = int(window_length / 4)
    threshold = db_to_amp(x=-trim_db, gain=gain, base=base)
    for x in range(hop_length, len(wav) - window_length, hop_length):
        if np.max(wav[x : x + window_length]) < threshold:
            return x + hop_length
    return len(wav)


def trim_silence(
    *,
    wav: np.ndarray = None,
    sample_rate: int = None,
    trim_db: float = None,
    win_length: int = None,
    hop_length: int = None,
    **kwargs,
) -> np.ndarray:
    """Trim silent parts with a threshold and 0.01 sec margin"""
    margin = int(sample_rate * 0.01)
    wav = wav[margin:-margin]
    return librosa.effects.trim(wav, top_db=trim_db, frame_length=win_length, hop_length=hop_length)[0]


def volume_norm(*, x: np.ndarray = None, coef: float = 0.95, **kwargs) -> np.ndarray:
    """Normalize the volume of an audio signal.

    Args:
        x (np.ndarray): Raw waveform.
        coef (float): Coefficient to rescale the maximum value. Defaults to 0.95.

    Returns:
        np.ndarray: Volume normalized waveform.
    """
    return x / abs(x).max() * coef


def rms_norm(*, wav: np.ndarray = None, db_level: float = -27.0, **kwargs) -> np.ndarray:
    r = 10 ** (db_level / 20)
    a = np.sqrt((len(wav) * (r**2)) / np.sum(wav**2))
    return wav * a


def rms_volume_norm(*, x: np.ndarray, db_level: float = -27.0, **kwargs) -> np.ndarray:
    """Normalize the volume based on RMS of the signal.

    Args:
        x (np.ndarray): Raw waveform.
        db_level (float): Target dB level in RMS. Defaults to -27.0.

    Returns:
        np.ndarray: RMS normalized waveform.
    """
    assert -99 <= db_level <= 0, " [!] db_level should be between -99 and 0"
    wav = rms_norm(wav=x, db_level=db_level)
    return wav


def load_wav(*, filename: str, sample_rate: int = None, resample: bool = False, **kwargs) -> np.ndarray:
    """Read a wav file using Librosa and optionally resample, silence trim, volume normalize.

    Resampling slows down loading the file significantly. Therefore it is recommended to resample the file before.

    Args:
        filename (str): Path to the wav file.
        sr (int, optional): Sampling rate for resampling. Defaults to None.
        resample (bool, optional): Resample the audio file when loading. Slows down the I/O time. Defaults to False.

    Returns:
        np.ndarray: Loaded waveform.
    """
    if resample:
        # loading with resampling. It is significantly slower.
        x, _ = librosa.load(filename, sr=sample_rate)
    else:
        # SF is faster than librosa for loading files
        x, _ = sf.read(filename)
    return x


def save_wav(*, wav: np.ndarray, path: str, sample_rate: int = None, **kwargs) -> None:
    """Save float waveform to a file using Scipy.

    Args:
        wav (np.ndarray): Waveform with float values in range [-1, 1] to save.
        path (str): Path to a output file.
        sr (int, optional): Sampling rate used for saving to the file. Defaults to None.
    """
    wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
    scipy.io.wavfile.write(path, sample_rate, wav_norm.astype(np.int16))


def mulaw_encode(*, wav: np.ndarray, mulaw_qc: int, **kwargs) -> np.ndarray:
    mu = 2**mulaw_qc - 1
    signal = np.sign(wav) * np.log(1 + mu * np.abs(wav)) / np.log(1.0 + mu)
    signal = (signal + 1) / 2 * mu + 0.5
    return np.floor(
        signal,
    )


def mulaw_decode(*, wav, mulaw_qc: int, **kwargs) -> np.ndarray:
    """Recovers waveform from quantized values."""
    mu = 2**mulaw_qc - 1
    x = np.sign(wav) / mu * ((1 + mu) ** np.abs(wav) - 1)
    return x


def encode_16bits(*, x: np.ndarray, **kwargs) -> np.ndarray:
    return np.clip(x * 2**15, -(2**15), 2**15 - 1).astype(np.int16)


def quantize(*, x: np.ndarray, quantize_bits: int, **kwargs) -> np.ndarray:
    """Quantize a waveform to a given number of bits.

    Args:
        x (np.ndarray): Waveform to quantize. Must be normalized into the range `[-1, 1]`.
        quantize_bits (int): Number of quantization bits.

    Returns:
        np.ndarray: Quantized waveform.
    """
    return (x + 1.0) * (2**quantize_bits - 1) / 2


def dequantize(*, x, quantize_bits, **kwargs) -> np.ndarray:
    """Dequantize a waveform from the given number of bits."""
    return 2 * x / (2**quantize_bits - 1) - 1
