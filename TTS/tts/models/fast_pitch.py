from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from coqpit import Coqpit
from matplotlib.pyplot import plot

from TTS.tts.layers.glow_tts.monotonic_align import generate_path, maximum_path
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.data import sequence_mask
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor

# pylint: disable=dangerous-default-value


class AlignmentEncoder(torch.nn.Module):
    """Module for alignment text and mel spectrogram."""

    def __init__(
        self,
        in_query_channels=80,
        in_key_channels=512,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_proj = nn.Sequential(
            ConvNorm(
                in_key_channels, in_key_channels * 2, kernel_size=3, bias=True, w_init_gain="relu", batch_norm=False
            ),
            torch.nn.ReLU(),
            ConvNorm(in_key_channels * 2, attn_channels, kernel_size=1, bias=True, batch_norm=False),
        )

        self.query_proj = nn.Sequential(
            ConvNorm(
                in_query_channels, in_query_channels * 2, kernel_size=3, bias=True, w_init_gain="relu", batch_norm=False
            ),
            torch.nn.ReLU(),
            ConvNorm(in_query_channels * 2, in_query_channels, kernel_size=1, bias=True, batch_norm=False),
            torch.nn.ReLU(),
            ConvNorm(in_query_channels, attn_channels, kernel_size=1, bias=True, batch_norm=False),
        )

    def forward(
        self, queries: torch.tensor, keys: torch.tensor, mask: torch.tensor = None, attn_prior: torch.tensor = None
    ):
        """Forward pass of the aligner encoder.
        Args:
            queries (torch.tensor): query tensor.
            keys (torch.tensor): key tensor.
            mask (torch.tensor): uint8 binary mask for variable length entries (should be in the T2 domain).
            attn_prior (torch.tensor): prior for attention matrix.
        Shapes:
            - queries: :math:`(B, C, T_de)`
            - keys: :math:`(B, C_emb, T_en)`
            - mask: :math:`(B, T_de)`
        Output:
            attn (torch.tensor): B x 1 x T1 x T2 attention mask. Final dim T2 should sum to 1.
            attn_logprob (torch.tensor): B x 1 x T1 x T2 log-prob attention mask.
        """
        keys_enc = self.key_proj(keys)  # B x n_attn_dims x T2
        queries_enc = self.query_proj(queries)

        # Simplistic Gaussian Isotopic Attention
        attn = (queries_enc[:, :, :, None] - keys_enc[:, :, None]) ** 2  # B x n_attn_dims x T1 x T2
        attn = -self.temperature * attn.sum(1, keepdim=True)

        if attn_prior is not None:
            attn = self.log_softmax(attn) + torch.log(attn_prior[:, None] + 1e-8)

        attn_logprob = attn.clone()

        if mask is not None:
            attn.data.masked_fill_(~mask.bool().unsqueeze(2), -float("inf"))

        attn = self.softmax(attn)  # softmax along T2
        return attn, attn_logprob


def mask_from_lens(lens, max_len: int = None):
    if max_len is None:
        max_len = lens.max()
    ids = torch.arange(0, max_len, device=lens.device, dtype=lens.dtype)
    mask = torch.lt(ids, lens.unsqueeze(1))
    return mask


class LinearNorm(torch.nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, w_init_gain="linear"):
        super(LinearNorm, self).__init__()
        self.linear_layer = torch.nn.Linear(in_dim, out_dim, bias=bias)

        torch.nn.init.xavier_uniform_(self.linear_layer.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class ConvNorm(torch.nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        padding=None,
        dilation=1,
        bias=True,
        w_init_gain="linear",
        batch_norm=False,
    ):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert kernel_size % 2 == 1
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=bias,
        )
        self.norm = torch.nn.BatchNorm1D(out_channels) if batch_norm else None

        torch.nn.init.xavier_uniform_(self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        if self.norm is None:
            return self.conv(signal)
        else:
            return self.norm(self.conv(signal))


class ConvReLUNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, dropout=0.0):
        super(ConvReLUNorm, self).__init__()
        self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=(kernel_size // 2))
        self.norm = torch.nn.LayerNorm(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, signal):
        out = F.relu(self.conv(signal))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2).to(signal.dtype)
        return self.dropout(out)


class PositionalEmbedding(nn.Module):
    def __init__(self, demb):
        super(PositionalEmbedding, self).__init__()
        self.demb = demb
        inv_freq = 1 / (10000 ** (torch.arange(0.0, demb, 2.0) / demb))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, pos_seq, bsz=None):
        sinusoid_inp = torch.matmul(torch.unsqueeze(pos_seq, -1), torch.unsqueeze(self.inv_freq, 0))
        pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=1)
        if bsz is not None:
            return pos_emb[None, :, :].expand(bsz, -1, -1)
        else:
            return pos_emb[None, :, :]


class PositionwiseConvFF(nn.Module):
    def __init__(self, d_model, d_inner, kernel_size, dropout, pre_lnorm=False):
        super(PositionwiseConvFF, self).__init__()

        self.d_model = d_model
        self.d_inner = d_inner
        self.dropout = dropout

        self.CoreNet = nn.Sequential(
            nn.Conv1d(d_model, d_inner, kernel_size, 1, (kernel_size // 2)),
            nn.ReLU(),
            # nn.Dropout(dropout),  # worse convergence
            nn.Conv1d(d_inner, d_model, kernel_size, 1, (kernel_size // 2)),
            nn.Dropout(dropout),
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.pre_lnorm = pre_lnorm

    def forward(self, inp):
        return self._forward(inp)

    def _forward(self, inp):
        if self.pre_lnorm:
            # layer normalization + positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(self.layer_norm(core_out).to(inp.dtype))
            core_out = core_out.transpose(1, 2)

            # residual connection
            output = core_out + inp
        else:
            # positionwise feed-forward
            core_out = inp.transpose(1, 2)
            core_out = self.CoreNet(core_out)
            core_out = core_out.transpose(1, 2)

            # residual connection + layer normalization
            output = self.layer_norm(inp + core_out).to(inp.dtype)

        return output


class MultiHeadAttn(nn.Module):
    def __init__(self, num_heads, d_model, hidden_channels_head, dropout, dropout_attn=0.1, pre_lnorm=False):
        super(MultiHeadAttn, self).__init__()

        self.num_heads = num_heads
        self.d_model = d_model
        self.hidden_channels_head = hidden_channels_head
        self.scale = 1 / (hidden_channels_head ** 0.5)
        self.pre_lnorm = pre_lnorm

        self.qkv_net = nn.Linear(d_model, 3 * num_heads * hidden_channels_head)
        self.drop = nn.Dropout(dropout)
        self.dropout_attn = nn.Dropout(dropout_attn)
        self.o_net = nn.Linear(num_heads * hidden_channels_head, d_model, bias=False)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp, attn_mask=None):
        return self._forward(inp, attn_mask)

    def _forward(self, inp, attn_mask=None):
        residual = inp

        if self.pre_lnorm:
            # layer normalization
            inp = self.layer_norm(inp)

        num_heads, hidden_channels_head = self.num_heads, self.hidden_channels_head

        head_q, head_k, head_v = torch.chunk(self.qkv_net(inp), 3, dim=2)
        head_q = head_q.view(inp.size(0), inp.size(1), num_heads, hidden_channels_head)
        head_k = head_k.view(inp.size(0), inp.size(1), num_heads, hidden_channels_head)
        head_v = head_v.view(inp.size(0), inp.size(1), num_heads, hidden_channels_head)

        q = head_q.permute(0, 2, 1, 3).reshape(-1, inp.size(1), hidden_channels_head)
        k = head_k.permute(0, 2, 1, 3).reshape(-1, inp.size(1), hidden_channels_head)
        v = head_v.permute(0, 2, 1, 3).reshape(-1, inp.size(1), hidden_channels_head)

        attn_score = torch.bmm(q, k.transpose(1, 2))
        attn_score.mul_(self.scale)

        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(1).to(attn_score.dtype)
            attn_mask = attn_mask.repeat(num_heads, attn_mask.size(2), 1)
            attn_score.masked_fill_(attn_mask.to(torch.bool), -float("inf"))

        attn_prob = F.softmax(attn_score, dim=2)
        attn_prob = self.dropout_attn(attn_prob)
        attn_vec = torch.bmm(attn_prob, v)

        attn_vec = attn_vec.view(num_heads, inp.size(0), inp.size(1), hidden_channels_head)
        attn_vec = (
            attn_vec.permute(1, 2, 0, 3).contiguous().view(inp.size(0), inp.size(1), num_heads * hidden_channels_head)
        )

        # linear projection
        attn_out = self.o_net(attn_vec)
        attn_out = self.drop(attn_out)

        if self.pre_lnorm:
            # residual connection
            output = residual + attn_out
        else:
            # residual connection + layer normalization
            output = self.layer_norm(residual + attn_out)

        output = output.to(attn_out.dtype)

        return output


class TransformerLayer(nn.Module):
    def __init__(
        self, num_heads, hidden_channels, hidden_channels_head, hidden_channels_ffn, kernel_size, dropout, **kwargs
    ):
        super(TransformerLayer, self).__init__()

        self.dec_attn = MultiHeadAttn(num_heads, hidden_channels, hidden_channels_head, dropout, **kwargs)
        self.pos_ff = PositionwiseConvFF(
            hidden_channels, hidden_channels_ffn, kernel_size, dropout, pre_lnorm=kwargs.get("pre_lnorm")
        )

    def forward(self, dec_inp, mask=None):
        output = self.dec_attn(dec_inp, attn_mask=~mask.squeeze(2))
        output *= mask
        output = self.pos_ff(output)
        output *= mask
        return output


class FFTransformer(nn.Module):
    def __init__(
        self,
        num_layers,
        num_heads,
        hidden_channels,
        hidden_channels_head,
        hidden_channels_ffn,
        kernel_size,
        dropout,
        dropout_attn,
        dropemb=0.0,
        pre_lnorm=False,
    ):
        super(FFTransformer, self).__init__()
        self.hidden_channels = hidden_channels
        self.num_heads = num_heads
        self.hidden_channels_head = hidden_channels_head

        self.pos_emb = PositionalEmbedding(self.hidden_channels)
        self.drop = nn.Dropout(dropemb)
        self.layers = nn.ModuleList()

        for _ in range(num_layers):
            self.layers.append(
                TransformerLayer(
                    num_heads,
                    hidden_channels,
                    hidden_channels_head,
                    hidden_channels_ffn,
                    kernel_size,
                    dropout,
                    dropout_attn=dropout_attn,
                    pre_lnorm=pre_lnorm,
                )
            )

    def forward(self, x, x_lengths, conditioning=0):
        mask = mask_from_lens(x_lengths).unsqueeze(2)

        pos_seq = torch.arange(x.size(1), device=x.device).to(x.dtype)
        pos_emb = self.pos_emb(pos_seq) * mask

        if conditioning is None:
            conditioning = 0

        out = self.drop(x + pos_emb + conditioning)

        for layer in self.layers:
            out = layer(out, mask=mask)

        # out = self.drop(out)
        return out, mask


def regulate_len(durations, enc_out, pace=1.0, mel_max_len=None):
    """If target=None, then predicted durations are applied"""
    dtype = enc_out.dtype
    reps = durations.float() / pace
    reps = (reps + 0.5).long()
    dec_lens = reps.sum(dim=1)

    max_len = dec_lens.max()
    reps_cumsum = torch.cumsum(F.pad(reps, (1, 0, 0, 0), value=0.0), dim=1)[:, None, :]
    reps_cumsum = reps_cumsum.to(dtype)

    range_ = torch.arange(max_len).to(enc_out.device)[None, :, None]
    mult = (reps_cumsum[:, :, :-1] <= range_) & (reps_cumsum[:, :, 1:] > range_)
    mult = mult.to(dtype)
    en_ex = torch.matmul(mult, enc_out)

    if mel_max_len:
        en_ex = en_ex[:, :mel_max_len]
        dec_lens = torch.clamp_max(dec_lens, mel_max_len)
    return en_ex, dec_lens


class TemporalPredictor(nn.Module):
    """Predicts a single float per each temporal location"""

    def __init__(self, input_size, filter_size, kernel_size, dropout, num_layers=2):
        super(TemporalPredictor, self).__init__()

        self.layers = nn.Sequential(
            *[
                ConvReLUNorm(
                    input_size if i == 0 else filter_size, filter_size, kernel_size=kernel_size, dropout=dropout
                )
                for i in range(num_layers)
            ]
        )
        self.fc = nn.Linear(filter_size, 1, bias=True)

    def forward(self, enc_out, enc_out_mask):
        out = enc_out * enc_out_mask
        out = self.layers(out.transpose(1, 2)).transpose(1, 2)
        out = self.fc(out) * enc_out_mask
        return out.squeeze(-1)


@dataclass
class FastPitchArgs(Coqpit):
    num_chars: int = 100
    out_channels: int = 80
    hidden_channels: int = 384
    num_speakers: int = 0
    duration_predictor_hidden_channels: int = 256
    duration_predictor_dropout: float = 0.1
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout_p: float = 0.1
    duration_predictor_num_layers: int = 2
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_dropout: float = 0.1
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    pitch_predictor_num_layers: int = 2
    positional_encoding: bool = True
    length_scale: int = 1
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {
            "hidden_channels_head": 64,
            "hidden_channels_ffn": 1536,
            "num_heads": 1,
            "num_layers": 6,
            "kernel_size": 3,
            "dropout": 0.1,
            "dropout_attn": 0.1,
        }
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {
            "hidden_channels_head": 64,
            "hidden_channels_ffn": 1536,
            "num_heads": 1,
            "num_layers": 6,
            "kernel_size": 3,
            "dropout": 0.1,
            "dropout_attn": 0.1,
        }
    )
    use_d_vector: bool = False
    d_vector_dim: int = 0
    detach_duration_predictor: bool = False
    max_duration: int = 75
    use_gt_duration: bool = True
    use_aligner: bool = True


class FastPitch(BaseTTS):
    """FastPitch model. Very similart to SpeedySpeech model but with pitch prediction.

    Paper abstract:
        We present FastPitch, a fully-parallel text-to-speech model based on FastSpeech, conditioned on fundamental
        frequency contours. The model predicts pitch contours during inference. By altering these predictions,
        the generated speech can be more expressive, better match the semantic of the utterance, and in the end
        more engaging to the listener. Uniformly increasing or decreasing pitch with FastPitch generates speech
        that resembles the voluntary modulation of voice. Conditioning on frequency contours improves the overall
        quality of synthesized speech, making it comparable to state-of-the-art. It does not introduce an overhead,
        and FastPitch retains the favorable, fully-parallel Transformer architecture, with over 900x real-time
        factor for mel-spectrogram synthesis of a typical utterance."

    Notes:
        TODO

    Args:
        config (Coqpit): Model coqpit class.

    Examples:
        >>> from TTS.tts.models.fast_pitch import FastPitch, FastPitchArgs
        >>> config = FastPitchArgs()
        >>> model = FastPitch(config)
    """

    def __init__(self, config: Coqpit):
        super().__init__()

        if "characters" in config:
            # loading from FasrPitchConfig
            _, self.config, num_chars = self.get_characters(config)
            config.model_args.num_chars = num_chars
            args = self.config.model_args
        else:
            # loading from FastPitchArgs
            self.config = config
            args = config

        self.max_duration = args.max_duration
        self.use_gt_duration = args.use_gt_duration
        self.use_aligner = args.use_aligner

        self.length_scale = float(args.length_scale) if isinstance(args.length_scale, int) else args.length_scale

        self.encoder = FFTransformer(
            hidden_channels=args.hidden_channels,
            **args.encoder_params,
        )

        # if n_speakers > 1:
        #     self.speaker_emb = nn.Embedding(n_speakers, symbols_embedding_dim)
        # else:
        #     self.speaker_emb = None
        # self.speaker_emb_weight = speaker_emb_weight
        self.emb = nn.Embedding(args.num_chars, args.hidden_channels)

        self.duration_predictor = TemporalPredictor(
            args.hidden_channels,
            filter_size=args.duration_predictor_hidden_channels,
            kernel_size=args.duration_predictor_kernel_size,
            dropout=args.duration_predictor_dropout_p,
            num_layers=args.duration_predictor_num_layers,
        )

        self.decoder = FFTransformer(hidden_channels=args.hidden_channels, **args.decoder_params)

        self.pitch_predictor = TemporalPredictor(
            args.hidden_channels,
            filter_size=args.pitch_predictor_hidden_channels,
            kernel_size=args.pitch_predictor_kernel_size,
            dropout=args.pitch_predictor_dropout_p,
            num_layers=args.pitch_predictor_num_layers,
        )

        self.pitch_emb = nn.Conv1d(
            1,
            args.hidden_channels,
            kernel_size=args.pitch_embedding_kernel_size,
            padding=int((args.pitch_embedding_kernel_size - 1) / 2),
        )

        self.proj = nn.Linear(args.hidden_channels, args.out_channels, bias=True)

        if args.use_aligner:
            self.aligner = AlignmentEncoder(args.out_channels, args.hidden_channels)

    @staticmethod
    def expand_encoder_outputs(en, dr, x_mask, y_mask):
        """Generate attention alignment map from durations and
        expand encoder outputs

        Example:
            encoder output: [a,b,c,d]
            durations: [1, 3, 2, 1]

            expanded: [a, b, b, b, c, c, d]
            attention map: [[0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 1, 0],
                            [0, 1, 1, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0]]
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(en.dtype)
        o_en_ex = torch.matmul(attn.transpose(1, 2), en)
        return o_en_ex, attn.transpose(1, 2)

    def forward(
        self, x, x_lengths, y_lengths, y=None, dr=None, pitch=None, aux_input={"d_vectors": 0, "speaker_ids": None}
    ):
        speaker_embedding = aux_input["d_vectors"] if "d_vectors" in aux_input else 0
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x.dtype)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        o_alignment_dur = None
        alignment_logprob = None
        alignment_mas = None

        # Calculate speaker embedding
        # if self.speaker_emb is None:
        #     speaker_embedding = 0
        # else:
        #     speaker_embedding = self.speaker_emb(speaker).unsqueeze(1)
        #     speaker_embedding.mul_(self.speaker_emb_weight)

        # character embedding
        embedding = self.emb(x)

        # Input FFT
        o_en, mask_en = self.encoder(embedding, x_lengths, conditioning=speaker_embedding)

        # Embedded for predictors
        o_en_dr, mask_en_dr = o_en, mask_en

        # Predict durations
        o_dr_log = self.duration_predictor(o_en_dr, mask_en_dr)
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)

        # Aligner
        if self.use_aligner:
            alignment_soft, alignment_logprob = self.aligner(y.transpose(1, 2), embedding.transpose(1, 2), x_mask, None)
            alignment_mas = maximum_path(
                alignment_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
            )
            o_alignment_dur = torch.log(1 + torch.sum(alignment_mas, -1))
            avg_pitch = average_pitch(pitch, o_alignment_dur)
            dr = o_alignment_dur

        # TODO: move this to the dataset
        avg_pitch = average_pitch(pitch, dr)

        # Predict pitch
        o_pitch = self.pitch_predictor(o_en, mask_en).unsqueeze(1)
        pitch_emb = self.pitch_emb(avg_pitch)
        o_en = o_en + pitch_emb.transpose(1, 2)

        # len_regulated, dec_lens = regulate_len(dr, o_en, self.length_scale, mel_max_len)
        o_en_ex, attn = self.expand_encoder_outputs(o_en, dr, x_mask, y_mask)

        # Output FFT
        o_de, _ = self.decoder(o_en_ex, y_lengths)
        o_de = self.proj(o_de)
        outputs = {
            "model_outputs": o_de,
            "durations_log": o_dr_log.squeeze(1),
            "durations": o_dr.squeeze(1),
            "pitch": o_pitch,
            "pitch_gt": avg_pitch,
            "alignments": attn,
            "alignment_mas": alignment_mas,
            "o_alignment_dur": o_alignment_dur,
            "alignment_logprob": alignment_logprob,
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": 0, "speaker_ids": None}):  # pylint: disable=unused-argument
        speaker_embedding = aux_input["d_vectors"] if "d_vectors" in aux_input else 0

        # input sequence should be greated than the max convolution size
        inference_padding = 5
        if x.shape[1] < 13:
            inference_padding += 13 - x.shape[1]

        # pad input to prevent dropping the last word
        x = torch.nn.functional.pad(x, pad=(0, inference_padding), mode="constant", value=0)
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)

        # character embedding
        embedding = self.emb(x)

        # if self.speaker_emb is None:
        # else:
        #     speaker = torch.ones(inputs.size(0)).long().to(inputs.device) * speaker
        #     spk_emb = self.speaker_emb(speaker).unsqueeze(1)
        #     spk_emb.mul_(self.speaker_emb_weight)

        # Input FFT
        o_en, mask_en = self.encoder(embedding, x_lengths, conditioning=speaker_embedding)

        # Predict durations
        o_dr_log = self.duration_predictor(o_en, mask_en)
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)
        o_dr = o_dr * self.length_scale

        # Pitch over chars
        o_pitch = self.pitch_predictor(o_en, mask_en).unsqueeze(1)

        # if pitch_transform is not None:
        #     if self.pitch_std[0] == 0.0:
        #         # XXX LJSpeech-1.1 defaults
        #         mean, std = 218.14, 67.24
        #     else:
        #         mean, std = self.pitch_mean[0], self.pitch_std[0]
        #     pitch_pred = pitch_transform(pitch_pred, mask_en.sum(dim=(1, 2)), mean, std)

        o_pitch_emb = self.pitch_emb(o_pitch).transpose(1, 2)

        o_en = o_en + o_pitch_emb

        y_lengths = o_dr.sum(1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype)
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x.dtype)

        o_en_ex, attn = self.expand_encoder_outputs(o_en, o_dr, x_mask, y_mask)
        o_de, _ = self.decoder(o_en_ex, y_lengths)
        o_de = self.proj(o_de)

        outputs = {"model_outputs": o_de, "alignments": attn, "pitch": o_pitch, "durations_log": o_dr_log}
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        pitch = batch["pitch"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        durations = batch["durations"]

        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        outputs = self.forward(
            text_input, text_lengths, mel_lengths, y=mel_input, dr=durations, pitch=pitch, aux_input=aux_input
        )

        if self.use_aligner:
            durations = outputs["o_alignment_dur"]

        # compute loss
        loss_dict = criterion(
            outputs["model_outputs"],
            mel_input,
            mel_lengths,
            outputs["durations_log"],
            durations,
            outputs["pitch"],
            outputs["pitch_gt"],
            text_lengths,
            outputs["alignment_logprob"],
        )

        # compute duration error
        durations_pred = outputs["durations"]
        duration_error = torch.abs(durations - durations_pred).sum() / text_lengths.sum()
        loss_dict["duration_error"] = duration_error
        return outputs, loss_dict

    def train_log(self, ap: AudioProcessor, batch: dict, outputs: dict):  # pylint: disable=no-self-use
        model_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        mel_input = batch["mel_input"]

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        if self.config.model_args.use_aligner and self.training:
            alignment_mas = outputs["alignment_mas"]
            figures["alignment_mas"] = plot_alignment(alignment_mas, ap, output_fig=False)

        # Sample audio
        train_audio = ap.inv_melspectrogram(pred_spec.T)
        return figures, {"audio": train_audio}

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, ap: AudioProcessor, batch: dict, outputs: dict):
        return self.train_log(ap, batch, outputs)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_criterion(self):
        from TTS.tts.layers.losses import FastPitchLoss  # pylint: disable=import-outside-toplevel

        return FastPitchLoss(self.config)


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg
