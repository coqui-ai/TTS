import torch
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class ConvBlockWithPrompting(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_attentions, attention_head, dropout):
        super().__init__()
        layers = []
        for i in range(n_layers):
            if i % 3 == 0 and i // 3 < n_attentions:
                layers.append(nn.MultiheadAttention(hidden_dim, attention_head))
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
                    nn.ReLU(),
                    Lambda(lambda x: x.transpose(1, 2)),
                    nn.LayerNorm(hidden_dim),
                    Lambda(lambda x: x.transpose(1, 2)),
                    nn.Dropout(dropout),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x, prompt_enc):
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                x = x.permute(2, 0, 1)
                prompt_enc_in = prompt_enc.permute(2, 0, 1)
                x, _ = layer(x, prompt_enc_in, prompt_enc_in)
                x = x.permute(1, 2, 0)
            else:
                x = layer(x)
        return x
