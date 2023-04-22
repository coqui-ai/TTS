from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import librosa
import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.nn import AvgPool1d, Conv1d, Conv2d, ConvTranspose1d
from torch.nn import functional as F
from torch.nn.utils import remove_weight_norm, spectral_norm, weight_norm

import TTS.vc.modules.freevc.commons as commons
import TTS.vc.modules.freevc.modules as modules
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.io import load_fsspec, save_checkpoint
from TTS.vc.configs.shared_configs import BaseVCConfig
from TTS.vc.models.base_vc import BaseVC
from TTS.vc.modules.freevc.commons import get_padding, init_weights
from TTS.vc.modules.freevc.mel_processing import mel_spectrogram_torch
from TTS.vc.modules.freevc.speaker_encoder.speaker_encoder import SpeakerEncoder as SpeakerEncoderEx
from TTS.vc.modules.freevc.wavlm import get_wavlm


class ResidualCouplingBlock(nn.Module):
    def __init__(self, channels, hidden_channels, kernel_size, dilation_rate, n_layers, n_flows=4, gin_channels=0):
        super().__init__()
        self.channels = channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.n_flows = n_flows
        self.gin_channels = gin_channels

        self.flows = nn.ModuleList()
        for i in range(n_flows):
            self.flows.append(
                modules.ResidualCouplingLayer(
                    channels,
                    hidden_channels,
                    kernel_size,
                    dilation_rate,
                    n_layers,
                    gin_channels=gin_channels,
                    mean_only=True,
                )
            )
            self.flows.append(modules.Flip())

    def forward(self, x, x_mask, g=None, reverse=False):
        if not reverse:
            for flow in self.flows:
                x, _ = flow(x, x_mask, g=g, reverse=reverse)
        else:
            for flow in reversed(self.flows):
                x = flow(x, x_mask, g=g, reverse=reverse)
        return x


class Encoder(nn.Module):
    def __init__(
        self, in_channels, out_channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.gin_channels = gin_channels

        self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
        self.enc = modules.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
        self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask


class Generator(torch.nn.Module):
    def __init__(
        self,
        initial_channel,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        gin_channels=0,
    ):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        self.conv_pre = Conv1d(initial_channel, upsample_initial_channel, 7, 1, padding=3)
        resblock = modules.ResBlock1 if resblock == "1" else modules.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        upsample_initial_channel // (2**i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        k,
                        u,
                        padding=(k - u) // 2,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)

    def forward(self, x, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.cond(g)

        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i * self.num_kernels + j](x)
                else:
                    xs += self.resblocks[i * self.num_kernels + j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print("Removing weight norm...")
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
                norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
            ]
        )
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList(
            [
                norm_f(Conv1d(1, 16, 15, 1, padding=7)),
                norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
                norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
                norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
                norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
                norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
            ]
        )
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2, 3, 5, 7, 11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class SpeakerEncoder(torch.nn.Module):
    def __init__(self, mel_n_channels=80, model_num_layers=3, model_hidden_size=256, model_embedding_size=256):
        super(SpeakerEncoder, self).__init__()
        self.lstm = nn.LSTM(mel_n_channels, model_hidden_size, model_num_layers, batch_first=True)
        self.linear = nn.Linear(model_hidden_size, model_embedding_size)
        self.relu = nn.ReLU()

    def forward(self, mels):
        self.lstm.flatten_parameters()
        _, (hidden, _) = self.lstm(mels)
        embeds_raw = self.relu(self.linear(hidden[-1]))
        return embeds_raw / torch.norm(embeds_raw, dim=1, keepdim=True)

    def compute_partial_slices(self, total_frames, partial_frames, partial_hop):
        mel_slices = []
        for i in range(0, total_frames - partial_frames, partial_hop):
            mel_range = torch.arange(i, i + partial_frames)
            mel_slices.append(mel_range)

        return mel_slices

    def embed_utterance(self, mel, partial_frames=128, partial_hop=64):
        mel_len = mel.size(1)
        last_mel = mel[:, -partial_frames:]

        if mel_len > partial_frames:
            mel_slices = self.compute_partial_slices(mel_len, partial_frames, partial_hop)
            mels = list(mel[:, s] for s in mel_slices)
            mels.append(last_mel)
            mels = torch.stack(tuple(mels), 0).squeeze(1)

            with torch.no_grad():
                partial_embeds = self(mels)
            embed = torch.mean(partial_embeds, axis=0).unsqueeze(0)
            # embed = embed / torch.linalg.norm(embed, 2)
        else:
            with torch.no_grad():
                embed = self(last_mel)

        return embed


@dataclass
class FreeVCAudioConfig(Coqpit):
    """Audio configuration

    Args:
        max_wav_value (float):
            The maximum value of the waveform.

        input_sample_rate (int):
            The sampling rate of the input waveform.

        output_sample_rate (int):
            The sampling rate of the output waveform.

        filter_length (int):
            The length of the filter.

        hop_length (int):
            The hop length.

        win_length (int):
            The window length.

        n_mel_channels (int):
            The number of mel channels.

        mel_fmin (float):
            The minimum frequency of the mel filterbank.

        mel_fmax (Optional[float]):
            The maximum frequency of the mel filterbank.
    """

    max_wav_value: float = field(default=32768.0)
    input_sample_rate: int = field(default=16000)
    output_sample_rate: int = field(default=24000)
    filter_length: int = field(default=1280)
    hop_length: int = field(default=320)
    win_length: int = field(default=1280)
    n_mel_channels: int = field(default=80)
    mel_fmin: float = field(default=0.0)
    mel_fmax: Optional[float] = field(default=None)


@dataclass
class FreeVCArgs(Coqpit):
    """FreeVC model arguments

    Args:
        spec_channels (int):
            The number of channels in the spectrogram.

        inter_channels (int):
            The number of channels in the intermediate layers.

        hidden_channels (int):
            The number of channels in the hidden layers.

        filter_channels (int):
            The number of channels in the filter layers.

        n_heads (int):
            The number of attention heads.

        n_layers (int):
            The number of layers.

        kernel_size (int):
            The size of the kernel.

        p_dropout (float):
            The dropout probability.

        resblock (str):
            The type of residual block.

        resblock_kernel_sizes (List[int]):
            The kernel sizes for the residual blocks.

        resblock_dilation_sizes (List[List[int]]):
            The dilation sizes for the residual blocks.

        upsample_rates (List[int]):
            The upsample rates.

        upsample_initial_channel (int):
            The number of channels in the initial upsample layer.

        upsample_kernel_sizes (List[int]):
            The kernel sizes for the upsample layers.

        n_layers_q (int):
            The number of layers in the quantization network.

        use_spectral_norm (bool):
            Whether to use spectral normalization.

        gin_channels (int):
            The number of channels in the global conditioning vector.

        ssl_dim (int):
            The dimension of the self-supervised learning embedding.

        use_spk (bool):
            Whether to use external speaker encoder.
    """

    spec_channels: int = field(default=641)
    inter_channels: int = field(default=192)
    hidden_channels: int = field(default=192)
    filter_channels: int = field(default=768)
    n_heads: int = field(default=2)
    n_layers: int = field(default=6)
    kernel_size: int = field(default=3)
    p_dropout: float = field(default=0.1)
    resblock: str = field(default="1")
    resblock_kernel_sizes: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates: List[int] = field(default_factory=lambda: [10, 8, 2, 2])
    upsample_initial_channel: int = field(default=512)
    upsample_kernel_sizes: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    n_layers_q: int = field(default=3)
    use_spectral_norm: bool = field(default=False)
    gin_channels: int = field(default=256)
    ssl_dim: int = field(default=1024)
    use_spk: bool = field(default=False)
    num_spks: int = field(default=0)
    segment_size: int = field(default=8960)


class FreeVC(BaseVC):
    """

    Papaer::
        https://arxiv.org/abs/2210.15418#

    Paper Abstract::
        Voice conversion (VC) can be achieved by first extracting source content information and target speaker
        information, and then reconstructing waveform with these information. However, current approaches normally
        either extract dirty content information with speaker information leaked in, or demand a large amount of
        annotated data for training. Besides, the quality of reconstructed waveform can be degraded by the
        mismatch between conversion model and vocoder. In this paper, we adopt the end-to-end framework of VITS for
        high-quality waveform reconstruction, and propose strategies for clean content information extraction without
        text annotation. We disentangle content information by imposing an information bottleneck to WavLM features,
        and propose the spectrogram-resize based data augmentation to improve the purity of extracted content
        information. Experimental results show that the proposed method outperforms the latest VC models trained with
        annotated data and has greater robustness.

    Original Code::
        https://github.com/OlaWod/FreeVC

    Examples:
        >>> from TTS.vc.configs.freevc_config import FreeVCConfig
        >>> from TTS.vc.models.freevc import FreeVC
        >>> config = FreeVCConfig()
        >>> model = FreeVC(config)
    """

    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager = None):
        super().__init__(config, None, speaker_manager, None)

        self.init_multispeaker(config)

        self.spec_channels = self.args.spec_channels
        self.inter_channels = self.args.inter_channels
        self.hidden_channels = self.args.hidden_channels
        self.filter_channels = self.args.filter_channels
        self.n_heads = self.args.n_heads
        self.n_layers = self.args.n_layers
        self.kernel_size = self.args.kernel_size
        self.p_dropout = self.args.p_dropout
        self.resblock = self.args.resblock
        self.resblock_kernel_sizes = self.args.resblock_kernel_sizes
        self.resblock_dilation_sizes = self.args.resblock_dilation_sizes
        self.upsample_rates = self.args.upsample_rates
        self.upsample_initial_channel = self.args.upsample_initial_channel
        self.upsample_kernel_sizes = self.args.upsample_kernel_sizes
        self.segment_size = self.args.segment_size
        self.gin_channels = self.args.gin_channels
        self.ssl_dim = self.args.ssl_dim
        self.use_spk = self.args.use_spk

        self.enc_p = Encoder(self.args.ssl_dim, self.inter_channels, self.hidden_channels, 5, 1, 16)
        self.dec = Generator(
            self.inter_channels,
            self.resblock,
            self.resblock_kernel_sizes,
            self.resblock_dilation_sizes,
            self.upsample_rates,
            self.upsample_initial_channel,
            self.upsample_kernel_sizes,
            gin_channels=self.gin_channels,
        )
        self.enc_q = Encoder(
            self.spec_channels, self.inter_channels, self.hidden_channels, 5, 1, 16, gin_channels=self.gin_channels
        )
        self.flow = ResidualCouplingBlock(
            self.inter_channels, self.hidden_channels, 5, 1, 4, gin_channels=self.gin_channels
        )
        if not self.use_spk:
            self.enc_spk = SpeakerEncoder(model_hidden_size=self.gin_channels, model_embedding_size=self.gin_channels)
        else:
            self.load_pretrained_speaker_encoder()

        self.wavlm = get_wavlm()

    @property
    def device(self):
        return next(self.parameters()).device

    def load_pretrained_speaker_encoder(self):
        """Load pretrained speaker encoder model as mentioned in the paper."""
        print(" > Loading pretrained speaker encoder model ...")
        self.enc_spk_ex = SpeakerEncoderEx(
            "https://github.com/coqui-ai/TTS/releases/download/v0.13.0_models/speaker_encoder.pt"
        )

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.num_spks = self.args.num_spks
        if self.speaker_manager:
            self.num_spks = self.speaker_manager.num_spks

    def forward(
        self,
        c: torch.Tensor,
        spec: torch.Tensor,
        g: Optional[torch.Tensor] = None,
        mel: Optional[torch.Tensor] = None,
        c_lengths: Optional[torch.Tensor] = None,
        spec_lengths: Optional[torch.Tensor] = None,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
    ]:
        """
        Forward pass of the model.

        Args:
            c: WavLM features. Shape: (batch_size, c_seq_len).
            spec: The input spectrogram. Shape: (batch_size, spec_seq_len, spec_dim).
            g: The speaker embedding. Shape: (batch_size, spk_emb_dim).
            mel: The input mel-spectrogram for the speaker encoder. Shape: (batch_size, mel_seq_len, mel_dim).
            c_lengths: The lengths of the WavLM features. Shape: (batch_size,).
            spec_lengths: The lengths of the spectrogram. Shape: (batch_size,).

        Returns:
            o: The output spectrogram. Shape: (batch_size, spec_seq_len, spec_dim).
            ids_slice: The slice indices. Shape: (batch_size, num_slices).
            spec_mask: The spectrogram mask. Shape: (batch_size, spec_seq_len).
            (z, z_p, m_p, logs_p, m_q, logs_q): A tuple of latent variables.
        """

        # If c_lengths is None, set it to the length of the last dimension of c
        if c_lengths is None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

        # If spec_lengths is None, set it to the length of the last dimension of spec
        if spec_lengths is None:
            spec_lengths = (torch.ones(spec.size(0)) * spec.size(-1)).to(spec.device)

        # If use_spk is False, compute g from mel using enc_spk
        g = None
        if not self.use_spk:
            g = self.enc_spk(mel).unsqueeze(-1)

        # Compute m_p, logs_p, z, m_q, logs_q, and spec_mask using enc_p and enc_q
        _, m_p, logs_p, _ = self.enc_p(c, c_lengths)
        z, m_q, logs_q, spec_mask = self.enc_q(spec.transpose(1, 2), spec_lengths, g=g)

        # Compute z_p using flow
        z_p = self.flow(z, spec_mask, g=g)

        # Randomly slice z and compute o using dec
        z_slice, ids_slice = commons.rand_slice_segments(z, spec_lengths, self.segment_size)
        o = self.dec(z_slice, g=g)

        return o, ids_slice, spec_mask, (z, z_p, m_p, logs_p, m_q, logs_q)

    @torch.no_grad()
    def inference(self, c, g=None, mel=None, c_lengths=None):
        """
        Inference pass of the model

        Args:
            c (torch.Tensor): Input tensor. Shape: (batch_size, c_seq_len).
            g (torch.Tensor): Speaker embedding tensor. Shape: (batch_size, spk_emb_dim).
            mel (torch.Tensor): Mel-spectrogram tensor. Shape: (batch_size, mel_seq_len, mel_dim).
            c_lengths (torch.Tensor): Lengths of the input tensor. Shape: (batch_size,).

        Returns:
            torch.Tensor: Output tensor.
        """
        if c_lengths == None:
            c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
        if not self.use_spk:
            g = self.enc_spk.embed_utterance(mel)
            g = g.unsqueeze(-1)
        z_p, m_p, logs_p, c_mask = self.enc_p(c, c_lengths)
        z = self.flow(z_p, c_mask, g=g, reverse=True)
        o = self.dec(z * c_mask, g=g)
        return o

    def extract_wavlm_features(self, y):
        """Extract WavLM features from an audio tensor.

        Args:
            y (torch.Tensor): Audio tensor. Shape: (batch_size, audio_seq_len).
        """

        with torch.no_grad():
            c = self.wavlm.extract_features(y)[0]
        c = c.transpose(1, 2)
        return c

    def load_audio(self, wav):
        """Read and format the input audio."""
        if isinstance(wav, str):
            wav, _ = librosa.load(wav, sr=self.config.audio.input_sample_rate)
        if isinstance(wav, np.ndarray):
            wav = torch.from_numpy(wav).to(self.device)
        if isinstance(wav, torch.Tensor):
            wav = wav.to(self.device)
        if isinstance(wav, list):
            wav = torch.from_numpy(np.array(wav)).to(self.device)
        return wav.float()

    @torch.inference_mode()
    def voice_conversion(self, src, tgt):
        """
        Voice conversion pass of the model.

        Args:
            src (str or torch.Tensor): Source utterance.
            tgt (str or torch.Tensor): Target utterance.

        Returns:
            torch.Tensor: Output tensor.
        """

        wav_tgt = self.load_audio(tgt).cpu().numpy()
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)

        if self.config.model_args.use_spk:
            g_tgt = self.enc_spk_ex.embed_utterance(wav_tgt)
            g_tgt = torch.from_numpy(g_tgt)[None, :, None].to(self.device)
        else:
            wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(self.device)
            mel_tgt = mel_spectrogram_torch(
                wav_tgt,
                self.config.audio.filter_length,
                self.config.audio.n_mel_channels,
                self.config.audio.input_sample_rate,
                self.config.audio.hop_length,
                self.config.audio.win_length,
                self.config.audio.mel_fmin,
                self.config.audio.mel_fmax,
            )
        # src
        wav_src = self.load_audio(src)
        c = self.extract_wavlm_features(wav_src[None, :])

        if self.config.model_args.use_spk:
            audio = self.inference(c, g=g_tgt)
        else:
            audio = self.inference(c, mel=mel_tgt.transpose(1, 2))
        audio = audio[0][0].data.cpu().float().numpy()
        return audio

    def eval_step():
        ...

    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        model = FreeVC(config)
        return model

    def load_checkpoint(self, config, checkpoint_path, eval=False, strict=True, cache=False):
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()

    def train_step():
        ...


@dataclass
class FreeVCConfig(BaseVCConfig):
    """Defines parameters for FreeVC End2End TTS model.

    Args:
        model (str):
            Model name. Do not change unless you know what you are doing.

        model_args (FreeVCArgs):
            Model architecture arguments. Defaults to `FreeVCArgs()`.

        audio (FreeVCAudioConfig):
            Audio processing configuration. Defaults to `FreeVCAudioConfig()`.

        grad_clip (List):
            Gradient clipping thresholds for each optimizer. Defaults to `[1000.0, 1000.0]`.

        lr_gen (float):
            Initial learning rate for the generator. Defaults to 0.0002.

        lr_disc (float):
            Initial learning rate for the discriminator. Defaults to 0.0002.

        lr_scheduler_gen (str):
            Name of the learning rate scheduler for the generator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_gen_params (dict):
            Parameters for the learning rate scheduler of the generator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`.

        lr_scheduler_disc (str):
            Name of the learning rate scheduler for the discriminator. One of the `torch.optim.lr_scheduler.*`. Defaults to
            `ExponentialLR`.

        lr_scheduler_disc_params (dict):
            Parameters for the learning rate scheduler of the discriminator. Defaults to `{'gamma': 0.999875, "last_epoch":-1}`.

        scheduler_after_epoch (bool):
            If true, step the schedulers after each epoch else after each step. Defaults to `False`.

        optimizer (str):
            Name of the optimizer to use with both the generator and the discriminator networks. One of the
            `torch.optim.*`. Defaults to `AdamW`.

        kl_loss_alpha (float):
            Loss weight for KL loss. Defaults to 1.0.

        disc_loss_alpha (float):
            Loss weight for the discriminator loss. Defaults to 1.0.

        gen_loss_alpha (float):
            Loss weight for the generator loss. Defaults to 1.0.

        feat_loss_alpha (float):
            Loss weight for the feature matching loss. Defaults to 1.0.

        mel_loss_alpha (float):
            Loss weight for the mel loss. Defaults to 45.0.

        return_wav (bool):
            If true, data loader returns the waveform as well as the other outputs. Do not change. Defaults to `True`.

        compute_linear_spec (bool):
            If true, the linear spectrogram is computed and returned alongside the mel output. Do not change. Defaults to `True`.

        use_weighted_sampler (bool):
            If true, use weighted sampler with bucketing for balancing samples between datasets used in training. Defaults to `False`.

        weighted_sampler_attrs (dict):
            Key retuned by the formatter to be used for weighted sampler. For example `{"root_path": 2.0, "speaker_name": 1.0}` sets sample probabilities
            by overweighting `root_path` by 2.0. Defaults to `{}`.

        weighted_sampler_multipliers (dict):
            Weight each unique value of a key returned by the formatter for weighted sampling.
            For example `{"root_path":{"/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-100/":1.0, "/raid/datasets/libritts-clean-16khz-bwe-coqui_44khz/LibriTTS/train-clean-360/": 0.5}`.
            It will sample instances from `train-clean-100` 2 times more than `train-clean-360`. Defaults to `{}`.

        r (int):
            Number of spectrogram frames to be generated at a time. Do not change. Defaults to `1`.

        add_blank (bool):
            If true, a blank token is added in between every character. Defaults to `True`.

        test_sentences (List[List]):
            List of sentences with speaker and language information to be used for testing.

        language_ids_file (str):
            Path to the language ids file.

        use_language_embedding (bool):
            If true, language embedding is used. Defaults to `False`.

    Note:
        Check :class:`TTS.tts.configs.shared_configs.BaseTTSConfig` for the inherited parameters.

    Example:

        >>> from TTS.tts.configs.freevc_config import FreeVCConfig
        >>> config = FreeVCConfig()
    """

    model: str = "freevc"
    # model specific params
    model_args: FreeVCArgs = FreeVCArgs()
    audio: FreeVCAudioConfig = FreeVCAudioConfig()

    # optimizer
    # TODO with training support

    # loss params
    # TODO with training support

    # data loader params
    return_wav: bool = True
    compute_linear_spec: bool = True

    # sampler params
    use_weighted_sampler: bool = False  # TODO: move it to the base config
    weighted_sampler_attrs: dict = field(default_factory=lambda: {})
    weighted_sampler_multipliers: dict = field(default_factory=lambda: {})

    # overrides
    r: int = 1  # DO NOT CHANGE
    add_blank: bool = True

    # multi-speaker settings
    # use speaker embedding layer
    num_speakers: int = 0
    speakers_file: str = None
    speaker_embedding_channels: int = 256

    # use d-vectors
    use_d_vector_file: bool = False
    d_vector_file: List[str] = None
    d_vector_dim: int = None

    def __post_init__(self):
        for key, val in self.model_args.items():
            if hasattr(self, key):
                self[key] = val
