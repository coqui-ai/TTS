import torch
import torch.nn as nn


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
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
        self.layers = nn.Sequential(*layers)

    def forward(self, x, prompt_enc):
        x = x.permute(2, 0, 1)
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                x, _ = layer(x, prompt_enc, prompt_enc)
            else:
                x = layer(x)
        x = x.permute(1, 2, 0)
        return x
