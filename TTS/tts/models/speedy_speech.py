import torch
from torch import nn
from TTS.tts.layers.speedy_speech.decoder import Decoder
from TTS.tts.layers.speedy_speech.duration_predictor import DurationPredictor
from TTS.tts.layers.speedy_speech.encoder import Encoder, PositionalEncoding
from TTS.tts.utils.generic_utils import sequence_mask
from TTS.tts.layers.glow_tts.monotonic_align import generate_path


class SpeedySpeech(nn.Module):
    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        num_chars,
        out_channels,
        hidden_channels,
        positional_encoding=True,
        length_scale=1,
        encoder_type='residual_conv_bn',
        encoder_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 13
        },
        decoder_residual_conv_bn_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4, 8] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 17
        },
        c_in_channels=0):
        super().__init__()
        self.length_scale = float(length_scale) if isinstance(length_scale, int) else length_scale
        self.emb = nn.Embedding(num_chars, hidden_channels)
        self.encoder = Encoder(hidden_channels, hidden_channels, encoder_type,
                               encoder_params, c_in_channels)
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_channels)
        self.decoder = Decoder(out_channels, hidden_channels,
                               decoder_residual_conv_bn_params)
        self.duration_predictor = DurationPredictor(hidden_channels)

    @staticmethod
    def expand_encoder_outputs(en, dr, x_mask, y_mask):
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(en.dtype)
        o_en_ex = torch.matmul(
            attn.squeeze(1).transpose(1, 2), en.transpose(1,
                                                          2)).transpose(1, 2)
        return o_en_ex, attn

    def format_durations(self, o_dr_log, x_mask):
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    def forward(self, x, x_lengths, y_lengths, dr, g=None):  # pylint: disable=unused-argument
        # TODO: multi-speaker
        # [B, T, C]
        x_emb = self.emb(x)
        # [B, C, T]
        x_emb = torch.transpose(x_emb, 1, -1)

        # compute sequence masks
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]),
                                 1).to(x.dtype)

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None),
                                 1).to(x_mask.dtype)

        # encoder pass
        o_en = self.encoder(x_emb, x_mask)

        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en.detach(), x_mask)

        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, dr, x_mask, y_mask)

        # positional encoding
        if hasattr(self, 'pos_encoder'):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)

        # decoder pass
        o_de = self.decoder(o_en_ex, y_mask)

        return o_de, o_dr_log.squeeze(1), attn.transpose(1, 2)

    def inference(self, x, x_lengths, g=None):  # pylint: disable=unused-argument
        # TODO: multi-speaker
        # pad input to prevent dropping the last word
        x = torch.nn.functional.pad(x, pad=(0, 5), mode='constant', value=0)

        # [B, T, C]
        x_emb = self.emb(x)
        # [B, C, T]
        x_emb = torch.transpose(x_emb, 1, -1)

        # compute sequence masks
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]),
                                 1).to(x.dtype)
        # encoder pass
        o_en = self.encoder(x_emb, x_mask)

        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en.detach(), x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)

        # output mask
        y_mask = torch.unsqueeze(sequence_mask(o_dr.sum(1), None), 1).to(x_mask.dtype)

        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, o_dr, x_mask, y_mask)

        # positional encoding
        if hasattr(self, 'pos_encoder'):
            o_en_ex = self.pos_encoder(o_en_ex)

        # decoder pass
        o_de = self.decoder(o_en_ex, y_mask)

        return o_de, attn.transpose(1, 2)
