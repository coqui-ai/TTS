import math
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.encoder import Encoder
from TTS.tts.layers.glow_tts.decoder import Decoder
from TTS.tts.utils.generic_utils import sequence_mask
from TTS.tts.layers.glow_tts.monotonic_align import maximum_path, generate_path


class GlowTts(nn.Module):
    """Glow TTS models from https://arxiv.org/abs/2005.11129"""
    def __init__(self,
                 num_chars,
                 hidden_channels,
                 filter_channels,
                 filter_channels_dp,
                 out_channels,
                 kernel_size=3,
                 num_heads=2,
                 num_layers_enc=6,
                 dropout_p=0.1,
                 num_flow_blocks_dec=12,
                 kernel_size_dec=5,
                 dilation_rate=5,
                 num_block_layers=4,
                 dropout_p_dec=0.,
                 num_speakers=0,
                 c_in_channels=0,
                 num_splits=4,
                 num_sqz=1,
                 sigmoid_scale=False,
                 rel_attn_window_size=None,
                 input_length=None,
                 mean_only=False,
                 hidden_channels_enc=None,
                 hidden_channels_dec=None,
                 use_encoder_prenet=False,
                 encoder_type="transformer",
                 external_speaker_embedding_dim=None):

        super().__init__()
        self.num_chars = num_chars
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.num_heads = num_heads
        self.num_layers_enc = num_layers_enc
        self.dropout_p = dropout_p
        self.num_flow_blocks_dec = num_flow_blocks_dec
        self.kernel_size_dec = kernel_size_dec
        self.dilation_rate = dilation_rate
        self.num_block_layers = num_block_layers
        self.dropout_p_dec = dropout_p_dec
        self.num_speakers = num_speakers
        self.c_in_channels = c_in_channels
        self.num_splits = num_splits
        self.num_sqz = num_sqz
        self.sigmoid_scale = sigmoid_scale
        self.rel_attn_window_size = rel_attn_window_size
        self.input_length = input_length
        self.mean_only = mean_only
        self.hidden_channels_enc = hidden_channels_enc
        self.hidden_channels_dec = hidden_channels_dec
        self.use_encoder_prenet = use_encoder_prenet
        self.noise_scale = 0.66
        self.length_scale = 1.
        self.external_speaker_embedding_dim = external_speaker_embedding_dim

        # if is a multispeaker and c_in_channels is 0, set to 256
        if num_speakers > 1:
            if self.c_in_channels == 0 and not self.external_speaker_embedding_dim:
                self.c_in_channels = 512
            elif self.external_speaker_embedding_dim:
                self.c_in_channels = self.external_speaker_embedding_dim

        self.encoder = Encoder(num_chars,
                               out_channels=out_channels,
                               hidden_channels=hidden_channels,
                               filter_channels=filter_channels,
                               filter_channels_dp=filter_channels_dp,
                               encoder_type=encoder_type,
                               num_heads=num_heads,
                               num_layers=num_layers_enc,
                               kernel_size=kernel_size,
                               dropout_p=dropout_p,
                               mean_only=mean_only,
                               use_prenet=use_encoder_prenet,
                               c_in_channels=self.c_in_channels)

        self.decoder = Decoder(out_channels,
                               hidden_channels_dec or hidden_channels,
                               kernel_size_dec,
                               dilation_rate,
                               num_flow_blocks_dec,
                               num_block_layers,
                               dropout_p=dropout_p_dec,
                               num_splits=num_splits,
                               num_sqz=num_sqz,
                               sigmoid_scale=sigmoid_scale,
                               c_in_channels=self.c_in_channels)

        if num_speakers > 1 and not external_speaker_embedding_dim:
            self.emb_g = nn.Embedding(num_speakers, self.c_in_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    @staticmethod
    def compute_outputs(attn, o_mean, o_log_scale, x_mask):
        # compute final values with the computed alignment
        y_mean = torch.matmul(
            attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)).transpose(
                1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        y_log_scale = torch.matmul(
            attn.squeeze(1).transpose(1, 2), o_log_scale.transpose(
                1, 2)).transpose(1, 2)  # [b, t', t], [b, t, d] -> [b, d, t']
        # compute total duration with adjustment
        o_attn_dur = torch.log(1 + torch.sum(attn, -1)) * x_mask
        return y_mean, y_log_scale, o_attn_dur

    def forward(self, x, x_lengths, y=None, y_lengths=None, attn=None, g=None):
        """
            Shapes:
                x: B x T
                x_lenghts: B
                y: B x C x T
                y_lengths: B
        """
        y_max_length = y.size(2)
        # norm speaker embeddings
        if g is not None:
            if self.external_speaker_embedding_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)# [b, h]

        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x,
                                                              x_lengths,
                                                              g=g)
        # format feature vectors and feature vector lenghts
        y, y_lengths, y_max_length, attn = self.preprocess(
            y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length),
                                 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)
        # find the alignment path
        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale,
                              [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 *
                                 (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2),
                                 z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale,
                              [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            attn = maximum_path(logp,
                                attn_mask.squeeze(1)).unsqueeze(1).detach()
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(
            attn, o_mean, o_log_scale, x_mask)
        attn = attn.squeeze(1).permute(0, 2, 1)
        return z, logdet, y_mean, y_log_scale, attn, o_dur_log, o_attn_dur

    @torch.no_grad()
    def inference(self, x, x_lengths, g=None):

        if g is not None:
            if self.external_speaker_embedding_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]

        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x,
                                                              x_lengths,
                                                              g=g)
        # compute output durations
        w = (torch.exp(o_dur_log) - 1) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None
        # compute masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length),
                                 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # compute attention mask
        attn = generate_path(w_ceil.squeeze(1),
                             attn_mask.squeeze(1)).unsqueeze(1)
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(
            attn, o_mean, o_log_scale, x_mask)
        z = (y_mean + torch.exp(y_log_scale) * torch.randn_like(y_mean) *
             self.noise_scale) * y_mask
        # decoder pass
        y, logdet = self.decoder(z, y_mask, g=g, reverse=True)
        attn = attn.squeeze(1).permute(0, 2, 1)
        return y, logdet, y_mean, y_log_scale, attn, o_dur_log, o_attn_dur

    def preprocess(self, y, y_lengths, y_max_length, attn=None):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.num_sqz) * self.num_sqz
            y = y[:, :, :y_max_length]
            if attn is not None:
                attn = attn[:, :, :, :y_max_length]
        y_lengths = (y_lengths // self.num_sqz) * self.num_sqz
        return y, y_lengths, y_max_length, attn

    def store_inverse(self):
        self.decoder.store_inverse()
