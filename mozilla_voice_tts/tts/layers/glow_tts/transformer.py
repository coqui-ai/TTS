import torch
from torch import nn
from torch.nn import functional as F

from mozilla_voice_tts.tts.layers.glow_tts.attention import MultiHeadAttention


class Transformer(nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 num_heads,
                 num_layers,
                 kernel_size=1,
                 dropout_p=0.,
                 rel_attn_window_size=None,
                 input_length=None,
                 **kwargs):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size
        self.input_length = input_length

        self.drop = nn.Dropout(dropout_p)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(
                MultiHeadAttention(hidden_channels,
                                   hidden_channels,
                                   num_heads,
                                   window_size=rel_attn_window_size,
                                   dropout_p=dropout_p,
                                   input_length=input_length))
            self.norm_layers_1.append(nn.GroupNorm(1, hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    dropout_p=dropout_p))
            self.norm_layers_2.append(nn.GroupNorm(1, hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x


class FFN(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 filter_channels,
                 kernel_size,
                 dropout_p=0.,
                 activation=None):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.activation = activation

        self.conv_1 = nn.Conv1d(in_channels,
                                filter_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels,
                                out_channels,
                                kernel_size,
                                padding=kernel_size // 2)
        self.drop = nn.Dropout(dropout_p)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask
