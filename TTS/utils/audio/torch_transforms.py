import librosa
import torch
from torch import nn


class TorchSTFT(nn.Module):  # pylint: disable=abstract-method
    """Some of the audio processing funtions using Torch for faster batch processing.

    Args:

        n_fft (int):
            FFT window size for STFT.

        hop_length (int):
            number of frames between STFT columns.

        win_length (int, optional):
            STFT window length.

        pad_wav (bool, optional):
            If True pad the audio with (n_fft - hop_length) / 2). Defaults to False.

        window (str, optional):
            The name of a function to create a window tensor that is applied/multiplied to each frame/window. Defaults to "hann_window"

        sample_rate (int, optional):
            target audio sampling rate. Defaults to None.

        mel_fmin (int, optional):
            minimum filter frequency for computing melspectrograms. Defaults to None.

        mel_fmax (int, optional):
            maximum filter frequency for computing melspectrograms. Defaults to None.

        n_mels (int, optional):
            number of melspectrogram dimensions. Defaults to None.

        use_mel (bool, optional):
            If True compute the melspectrograms otherwise. Defaults to False.

        do_amp_to_db_linear (bool, optional):
            enable/disable amplitude to dB conversion of linear spectrograms. Defaults to False.

        spec_gain (float, optional):
            gain applied when converting amplitude to DB. Defaults to 1.0.

        power (float, optional):
            Exponent for the magnitude spectrogram, e.g., 1 for energy, 2 for power, etc.  Defaults to None.

        use_htk (bool, optional):
            Use HTK formula in mel filter instead of Slaney.

        mel_norm (None, 'slaney', or number, optional):
            If 'slaney', divide the triangular mel weights by the width of the mel band
            (area normalization).

            If numeric, use `librosa.util.normalize` to normalize each filter by to unit l_p norm.
            See `librosa.util.normalize` for a full description of supported norm values
            (including `+-np.inf`).

            Otherwise, leave all the triangles aiming for a peak value of 1.0. Defaults to "slaney".
    """

    def __init__(
        self,
        n_fft,
        hop_length,
        win_length,
        pad_wav=False,
        window="hann_window",
        sample_rate=None,
        mel_fmin=0,
        mel_fmax=None,
        n_mels=80,
        use_mel=False,
        do_amp_to_db=False,
        spec_gain=1.0,
        power=None,
        use_htk=False,
        mel_norm="slaney",
        normalized=False,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.pad_wav = pad_wav
        self.sample_rate = sample_rate
        self.mel_fmin = mel_fmin
        self.mel_fmax = mel_fmax
        self.n_mels = n_mels
        self.use_mel = use_mel
        self.do_amp_to_db = do_amp_to_db
        self.spec_gain = spec_gain
        self.power = power
        self.use_htk = use_htk
        self.mel_norm = mel_norm
        self.window = nn.Parameter(getattr(torch, window)(win_length), requires_grad=False)
        self.mel_basis = None
        self.normalized = normalized
        if use_mel:
            self._build_mel_basis()

    def __call__(self, x):
        """Compute spectrogram frames by torch based stft.

        Args:
            x (Tensor): input waveform

        Returns:
            Tensor: spectrogram frames.

        Shapes:
            x: [B x T] or [:math:`[B, 1, T]`]
        """
        if x.ndim == 2:
            x = x.unsqueeze(1)
        if self.pad_wav:
            padding = int((self.n_fft - self.hop_length) / 2)
            x = torch.nn.functional.pad(x, (padding, padding), mode="reflect")
        # B x D x T x 2
        o = torch.stft(
            x.squeeze(1),
            self.n_fft,
            self.hop_length,
            self.win_length,
            self.window,
            center=True,
            pad_mode="reflect",  # compatible with audio.py
            normalized=self.normalized,
            onesided=True,
            return_complex=False,
        )
        M = o[:, :, :, 0]
        P = o[:, :, :, 1]
        S = torch.sqrt(torch.clamp(M**2 + P**2, min=1e-8))

        if self.power is not None:
            S = S**self.power

        if self.use_mel:
            S = torch.matmul(self.mel_basis.to(x), S)
        if self.do_amp_to_db:
            S = self._amp_to_db(S, spec_gain=self.spec_gain)
        return S

    def _build_mel_basis(self):
        mel_basis = librosa.filters.mel(
            sr=self.sample_rate,
            n_fft=self.n_fft,
            n_mels=self.n_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax,
            htk=self.use_htk,
            norm=self.mel_norm,
        )
        self.mel_basis = torch.from_numpy(mel_basis).float()

    @staticmethod
    def _amp_to_db(x, spec_gain=1.0):
        return torch.log(torch.clamp(x, min=1e-5) * spec_gain)

    @staticmethod
    def _db_to_amp(x, spec_gain=1.0):
        return torch.exp(x) / spec_gain
