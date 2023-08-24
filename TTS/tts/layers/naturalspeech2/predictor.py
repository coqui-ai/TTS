import torch
import torch.nn as nn
import math
from flash_attn.modules.mha import MHA, ParallelMHA
def generate_causal_mask(seq_len, device):
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
    return mask

class Lambda(nn.Module):
    def __init__(self, lambd):
        super().__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)
class ConvTBC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding=0):
        super(ConvTBC, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding

        self.weight = torch.nn.Parameter(torch.Tensor(
            self.kernel_size, in_channels, out_channels))
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))

    def forward(self, input):
        return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding)

class EncConvLayer(nn.Module):
    def __init__(self, c, kernel_size, dropout):
        super().__init__()
        self.layer_norm = LayerNorm(c)
        self.conv = ConvTBC(c, c, kernel_size, padding=kernel_size // 2)
        std = math.sqrt((4 * (1.0 - dropout)) / (kernel_size * c))
        nn.init.normal_(conv.weight, mean=0, std=std)
        nn.init.constant_(conv.bias, 0)
        # self.conv = LayerNorm(conv)
        self.dropout = dropout

    def forward(self, x, encoder_padding_mask=None, **kwargs):
        layer_norm_training = kwargs.get('layer_norm_training', None)
        if layer_norm_training is not None:
            self.layer_norm.training = layer_norm_training
        residual = x
        if encoder_padding_mask is not None:
            x = x.masked_fill(encoder_padding_mask.t().unsqueeze(-1), 0)
        x = self.layer_norm(x)
        x = self.conv(x)
        x = torch.functional.relu(x)
        x = torch.functional.dropout(x, self.dropout, self.training)
        x = x + residual
        return x

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
# class RMSNorm(nn.Module):
#     def __init__(self, dim, scale = True, dim_cond = None):
#         super().__init__()
#         self.to_gamma_beta = None
#         if dim_cond is not None:
#             self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

#         self.scale = dim ** 0.5
#         self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

#     def forward(self, x, cond = None):
#         out = nn.functional.normalize(x, dim = -1) * self.scale * self.gamma
        
#         if self.to_gamma_beta is None:
#             return out

#         gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
#         gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (self.gamma, beta))
#         return out * gamma + beta
        
class ConvBlockWithPrompting(nn.Module):
    def __init__(self, hidden_dim, n_layers, n_attentions, attention_head, kernel_size, dropout, task, is_causal=True):
        super().__init__()
        layers = []
        att_at = n_layers // n_attentions
        for i in range(n_layers):
            if (i+1) % att_at == 0:
                print("MultiheadAttention added at ", i)
                layers.append(MHA(hidden_dim,
                attention_head,
                num_heads_kv=attention_head,
                cross_attn=True,
                dropout=dropout,
                causal=True,
                fused_bias_fc=False,
                use_flash_attn=False))
                # layers.append(nn.MultiheadAttention(hidden_dim, attention_head, dropout))
            layers.extend(
                [
                    nn.Conv1d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=1),
                    nn.SiLU(),
                    LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
        self.layers = nn.Sequential(*layers)
        self.proj = nn.Conv1d(in_channels= hidden_dim, out_channels= 1, kernel_size=kernel_size, padding=1)
        self.activation = nn.ReLU()
        self.task = task
        self.is_causal = is_causal
    def forward(self, x, prompt_enc, mask=None):
        for i, layer in enumerate(self.layers):
            residual = x
            if isinstance(layer, MHA):
                # x = x.permute(2, 0, 1)
                # attn_mask=None
                # if self.is_causal:
                #     seq_len = x.size(0)
                #     attn_mask = generate_causal_mask(seq_len, x.device)
                # prompt_enc_in = prompt_enc.permute(2, 0, 1)
                # print(x.shape, prompt_enc.shape)
                x = layer(x.transpose(1,2), prompt_enc.transpose(1,2))
                x = x.transpose(1,2)
            else:
                if not mask is None:
                    x = layer(x * mask)
                else:
                    x = layer(x)
            if i % 2 == 1:  # if it's an even layer (since we start from 0), we add the residual connection
                x = x + residual
        if not mask is None:
            x = self.proj(x) * mask
        else:
            x = self.proj(x)
        if self.task != "pitch":
            x = self.activation(x)
        return x