import torch
import torch.nn as nn


class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-4):
        """Layer norm for the 2nd dimension of the input.
        Args:
            channels (int): number of channels (2nd dimension) of the input.
            eps (float): to prevent 0 division

        Shapes:
            - input: (B, C, T)
            - output: (B, C, T)
        """
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(1, channels, 1) * 0.1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1))

    def forward(self, x):
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        x = x * self.gamma + self.beta
        return x
class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.to_gamma_beta = None
        if dim_cond is not None:
            self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        out = nn.functional.normalize(x, dim = -1) * self.scale * self.gamma
        
        if self.to_gamma_beta is None:
            return out

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (self.gamma, beta))
        return out * gamma + beta
        
class ConvBlockWithPrompting(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_attentions, attention_head, kernel_size, dropout, task):
        super().__init__()
        layers = []
        att_at = n_layers // n_attentions
        for i in range(n_layers):
            if i % att_at == 0:
                layers.append(nn.MultiheadAttention(hidden_dim, attention_head, dropout))
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1),
                    nn.ReLU(),
                    LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Conv1d(in_channels= hidden_dim, out_channels= 1, kernel_size=kernel_size, padding=1)
        self.activation = nn.ReLU()
        self.task = task
    def forward(self, x, prompt_enc, mask=None):
        for layer in self.layers:
            if isinstance(layer, nn.MultiheadAttention):
                x = x.permute(2, 0, 1)
                prompt_enc_in = prompt_enc.permute(2, 0, 1)
                x, _ = layer(x, prompt_enc_in, prompt_enc_in)
                x = x.permute(1, 2, 0)
            else:
                if not mask is None:
                    x = layer(x * mask)
                else:
                    x = layer(x)
        if not mask is None:
            x = self.proj(x) * mask
        else:
            x = self.proj(x)
        if self.task != "pitch":
            x = self.activation(x)
        return x