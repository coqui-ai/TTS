from typing import Optional, Union

import torch
import torch.nn as nn

from TTS.tts.utils.helpers import sequence_mask
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()

        self.pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :].to(x.device)
        return x

class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model,
        nhead,
        num_layers,
        dim_feedforward,
        kernel_size,
        dropout,
        language_emb_dim=4,
        max_len=250,
        n_vocab=100,
        encoder_type="phoneme",
    ):
        super().__init__()
        if encoder_type == "phoneme":
            self.pre = nn.Embedding(n_vocab, d_model)
        else:
            self.pre = nn.Conv1d(128, d_model, kernel_size, padding=kernel_size // 2)
        # self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers
        )
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, dim_feedforward, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(dim_feedforward, d_model, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x, x_mask=None,lang_emb=None):
        if x_mask is None:
            x_len = torch.tensor(x.shape[2:]).to(x.device)
            x_mask = torch.unsqueeze(sequence_mask(x_len, None), 1).float()
        x = self.pre(x)
        if x.dim() == 4:
            x = x.squeeze(2)
        else:
            x = x.transpose(1, 2)
        # x = self.pos_enc(x * x_mask.transpose(1,2))
        x = self.transformer(x * x_mask.transpose(1,2))
        x = x.transpose(1, 2)
        x = self.conv(x * x_mask)
        x = x.transpose(1, 2)
        return x
