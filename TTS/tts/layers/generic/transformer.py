import torch
import torch.nn.functional as F
from torch import nn


class FFTransformer(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn=1024, kernel_size_fft=3, dropout_p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_out_channels, num_heads, dropout=dropout_p)

        padding = (kernel_size_fft - 1) // 2
        self.conv1 = nn.Conv1d(in_out_channels, hidden_channels_ffn, kernel_size=kernel_size_fft, padding=padding)
        self.conv2 = nn.Conv1d(hidden_channels_ffn, in_out_channels, kernel_size=kernel_size_fft, padding=padding)

        self.norm1 = nn.LayerNorm(in_out_channels)
        self.norm2 = nn.LayerNorm(in_out_channels)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """😦 ugly looking with all the transposing"""
        src = src.permute(2, 0, 1)
        src2, enc_align = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src + src2)
        # T x B x D -> B x D x T
        src = src.permute(1, 2, 0)
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = self.dropout2(src2)
        src = src + src2
        src = src.transpose(1, 2)
        src = self.norm2(src)
        src = src.transpose(1, 2)
        return src, enc_align


class FFTransformerBlock(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn, num_layers, dropout_p):
        super().__init__()
        self.fft_layers = nn.ModuleList(
            [
                FFTransformer(
                    in_out_channels=in_out_channels,
                    num_heads=num_heads,
                    hidden_channels_ffn=hidden_channels_ffn,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, g=None):  # pylint: disable=unused-argument
        """
        TODO: handle multi-speaker
        Shapes:
            - x: :math:`[B, C, T]`
            - mask:  :math:`[B, 1, T] or [B, T]`
        """
        if mask is not None and mask.ndim == 3:
            mask = mask.squeeze(1)
            # mask is negated, torch uses 1s and 0s reversely.
            mask = ~mask.bool()
        alignments = []
        for layer in self.fft_layers:
            x, align = layer(x, src_key_padding_mask=mask)
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)
        return x


class FFTDurationPredictor:
    def __init__(
        self, in_channels, hidden_channels, num_heads, num_layers, dropout_p=0.1, cond_channels=None
    ):  # pylint: disable=unused-argument
        self.fft = FFTransformerBlock(in_channels, num_heads, hidden_channels, num_layers, dropout_p)
        self.proj = nn.Linear(in_channels, 1)

    def forward(self, x, mask=None, g=None):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, C, T]`
            - mask:  :math:`[B, 1, T]`

        TODO: Handle the cond input
        """
        x = self.fft(x, mask=mask)
        x = self.proj(x)
        return x
