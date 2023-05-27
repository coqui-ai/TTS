import torch
import torch.nn as nn
from typing import Optional, Union

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
        x = x + self.pe[: x.size(0), :]
        return x


class TransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead, num_layers, dim_feedforward, kernel_size, dropout, language_emb_dim = 4, max_len=250):
        super().__init__()
        if language_emb_dim:
            d_model += language_emb_dim
        self.embedding = nn.Embedding(d_model, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_len)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout), num_layers
        )
        self.conv = nn.Sequential(
            nn.Conv1d(d_model, dim_feedforward, kernel_size, padding=kernel_size // 2),
            nn.ReLU(),
            nn.Conv1d(dim_feedforward, d_model, kernel_size, padding=kernel_size // 2),
        )

    def forward(self, x, lang_emb):
        x = self.embedding(x)
        # concat the lang emb in embedding chars
        if lang_emb is not None:
            x = torch.cat((x, lang_emb.transpose(2, 1).expand(x.size(0), x.size(1), -1)), dim=-1)
        x = self.pos_enc(x)
        x = self.transformer(x)
        x = x.transpose(1, 2)
        x = self.conv(x)
        x = x.transpose(1, 2)
        return x