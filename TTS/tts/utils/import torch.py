import torch
import librosa
import soundfile as sf
import numpy as np
import scipy.io
import scipy.signal

from TTS.tts.utils.stft_torch import STFT

class AudioProcessor(object):
    def __init__(self,
                 sample_rate=None,
                 num_mels=None,
                 frame_shift_ms=None,
                 frame_length_ms=None,
                 hop_length=None,
                 win_length=None,
                 num_freq=None,
                 power=None,
                 mel_fmin=None,
                 mel_fmax=None,
                 griffin_lim_iters=None,
                 do_trim_silence=False,
                 trim_db=60,
                 sound_norm=False,
                 use_cuda=False,
                 **_):

        print(" > Setting up Torch based Audio Processor...")
        # setup class attributed
        self.sample_rate = sample_rate
        self.num_mels = num_mels
        self.frame_shift_ms = frame_shift_ms
        self.frame_length_ms = frame_length_ms
        self.num_freq = num_freq
        self.power = power
        self.griffin_lim_iters = griffin_lim_iters
        self.mel_fmin = mel_fmin or 0
        self.mel_fmax = mel_fmax
        self.do_trim_silence = do_trim_silence
        self.trim_db = trim_db
        self.sound_norm = sound_norm
        # setup stft parameters
        if hop_length is None:
            self.n_fft, self.hop_length, self.win_length = self._stft_parameters()
        else:
            self.hop_length = hop_length
            self.win_length = win_length
            self.n_fft = (self.num_freq - 1) * 2
        members = vars(self)
        # print class attributes
        for key, value in members.items():
            print(" | > {}:{}".format(key, value))
        # create spectrogram utils
        self.mel_basis = torch.from_numpy(self._build_mel_basis()).float()
        self.inv_mel_basis = torch.from_numpy(np.linalg.pinv(self._build_mel_basis())).float()
        self.stft = STFT(filter_length=self.n_fft, hop_length=self.hop_length, win_length=self.win_length,
                         window='hann', padding_mode='constant', use_cuda=use_cuda)

    ### setting up the parameters ###
    def _build_mel_basis(self):
        if self.mel_fmax is not None:
            assert self.mel_fmax <= self.sample_rate // 2
        return librosa.filters.mel(
            self.sample_rate,
            self.n_fft,
            n_mels=self.num_mels,
            fmin=self.mel_fmin,
            fmax=self.mel_fmax)

    def _stft_parameters(self, ):
        """Compute necessary stft parameters with given time values"""
        n_fft = (self.num_freq - 1) * 2
        factor = self.frame_length_ms / self.frame_shift_ms
        assert (factor).is_integer(), " [!] frame_shift_ms should divide frame_length_ms"
        hop_length = int(self.frame_shift_ms / 1000.0 * self.sample_rate)
        win_length = int(hop_length * factor)
        return n_fft, hop_length, win_length

    ### DB and AMP conversion ###
    def amp_to_db(self, x):
        return torch.log10(torch.clamp(x, min=1e-5))

    def db_to_amp(self, x):
        return torch.pow(10.0, x)

    ### SPECTROGRAM ###
    def linear_to_mel(self, spectrogram):
        return torch.matmul(self.mel_basis, spectrogram)

    def mel_to_linear(self, mel_spec):
        return np.maximum(1e-10, np.matmul(self.inv_mel_basis, mel_spec))

    def spectrogram(self, y):
        ''' Compute spectrograms 
        Args:
            y (Tensor): audio signal. (B x T)
        '''
        M, P = self.stft.transform(y)
        return self.amp_to_db(M)

    def melspectrogram(self, y):
        ''' Compute mel-spectrograms 
        Args:
            y (Tensor): audio signal. (B x T)
        '''
        M, P = self.stft.transform(y)
        return self.amp_to_db(self.linear_to_mel(M))

    ### INV SPECTROGRAM ###
    def inv_spectrogram(self, S):
        """Converts spectrogram to waveform using librosa"""
        S = self.db_to_amp(S)
        return self.griffin_lim(S**self.power)

    def inv_melspectrogram(self, S):
        '''Converts mel spectrogram to waveform using librosa'''
        S = self.db_to_amp(S)
        S = self.mel_to_linear(S)  # Convert back to linear
        return self.griffin_lim(S**self.power)

    def out_linear_to_mel(self, linear_spec):
        S = self._denormalize(linear_spec)
        S = self._db_to_amp(S)
        S = self._linear_to_mel(np.abs(S))
        S = self._amp_to_db(S)
        mel = self._normalize(S)
        return mel

    def griffin_lim(self, S):
        """
	    PARAMS
	    ------
	    magnitudes: spectrogram magnitudes
	    """

        angles = np.angle(np.exp(2j * np.pi * np.random.rand(*S.size())))
        angles = angles.astype(np.float32)
        angles = torch.from_numpy(angles)
        signal = self.stft.inverse(S, angles).squeeze(1)

        for _ in range(self.griffin_lim_iters):
            _, angles = self.stft.transform(signal)
            signal = self.stft.inverse(S, angles).squeeze(1)
        return signal

    ### Audio processing ###
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

    def sound_norm(self, x):
        return x / abs(x).max() * 0.9

    ### SAVE and LOAD ###
    def load_wav(self, filename, sr=None):
        if sr is None:
            x, sr = sf.read(filename)
        else:
            x, sr = librosa.load(filename, sr=sr)
        return x
    
    def save_wav(self, wav, path):
        wav_norm = wav * (32767 / max(0.01, np.max(np.abs(wav))))
        scipy.io.wavfile.write(path, self.sample_rate, wav_norm.astype(np.int16))