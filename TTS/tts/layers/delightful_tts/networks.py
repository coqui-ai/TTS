import math
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F

from TTS.tts.layers.delightful_tts.conv_layers import ConvNorm


def initialize_embeddings(shape: Tuple[int]) -> torch.Tensor:
    assert len(shape) == 2, "Can only initialize 2-D embedding matrices ..."
    # Kaiming initialization
    return torch.randn(shape) * np.sqrt(2 / shape[1])


def positional_encoding(d_model: int, length: int, device: torch.device) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)
    return pe


class BottleneckLayer(nn.Module):
    """
    Bottleneck layer for reducing the dimensionality of a tensor.

    Args:
        in_dim: The number of input dimensions.
        reduction_factor: The factor by which to reduce the number of dimensions.
        norm: The normalization method to use. Can be "weightnorm" or "instancenorm".
        non_linearity: The non-linearity to use. Can be "relu" or "leakyrelu".
        kernel_size: The size of the convolutional kernel.
        use_partial_padding: Whether to use partial padding with the convolutional kernel.

    Shape:
        - Input: :math:`[N, in_dim]` where `N` is the batch size and `in_dim` is the number of input dimensions.

        - Output: :math:`[N, out_dim]` where `out_dim` is the number of output dimensions.
    """

    def __init__(
        self,
        in_dim,
        reduction_factor,
        norm="weightnorm",
        non_linearity="relu",
        kernel_size=3,
        use_partial_padding=False,  # pylint: disable=unused-argument
    ):
        super(BottleneckLayer, self).__init__()  # pylint: disable=super-with-arguments

        self.reduction_factor = reduction_factor
        reduced_dim = int(in_dim / reduction_factor)
        self.out_dim = reduced_dim
        if self.reduction_factor > 1:
            fn = ConvNorm(in_dim, reduced_dim, kernel_size=kernel_size, use_weight_norm=(norm == "weightnorm"))
            if norm == "instancenorm":
                fn = nn.Sequential(fn, nn.InstanceNorm1d(reduced_dim, affine=True))

            self.projection_fn = fn
            self.non_linearity = nn.ReLU()
            if non_linearity == "leakyrelu":
                self.non_linearity = nn.LeakyReLU()

    def forward(self, x):
        if self.reduction_factor > 1:
            x = self.projection_fn(x)
            x = self.non_linearity(x)
        return x


class GLUActivation(nn.Module):
    """Class that implements the Gated Linear Unit (GLU) activation function.

    The GLU activation function is a variant of the Leaky ReLU activation function,
    where the output of the activation function is gated by an input tensor.

    """

    def __init__(self, slope: float):
        super().__init__()
        self.lrelu = nn.LeakyReLU(slope)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, gate = x.chunk(2, dim=1)
        x = out * self.lrelu(gate)
        return x


class StyleEmbedAttention(nn.Module):
    def __init__(self, query_dim: int, key_dim: int, num_units: int, num_heads: int):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query: torch.Tensor, key_soft: torch.Tensor) -> torch.Tensor:
        values = self.W_value(key_soft)
        split_size = self.num_units // self.num_heads
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)

        out_soft = scores_soft = None
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key_soft)  # [N, T_k, num_units]

        # [h, N, T_q, num_units/h]
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)
        # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores_soft = torch.matmul(querys, keys.transpose(2, 3))  # [h, N, T_q, T_k]
        scores_soft = scores_soft / (self.key_dim**0.5)
        scores_soft = F.softmax(scores_soft, dim=3)

        # out = score * V
        # [h, N, T_q, num_units/h]
        out_soft = torch.matmul(scores_soft, values)
        out_soft = torch.cat(torch.split(out_soft, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out_soft  # , scores_soft


class EmbeddingPadded(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, padding_idx: int):
        super().__init__()
        padding_mult = torch.ones((num_embeddings, 1), dtype=torch.int64)
        padding_mult[padding_idx] = 0
        self.register_buffer("padding_mult", padding_mult)
        self.embeddings = nn.parameter.Parameter(initialize_embeddings((num_embeddings, embedding_dim)))

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        embeddings_zeroed = self.embeddings * self.padding_mult
        x = F.embedding(idx, embeddings_zeroed)
        return x


class EmbeddingProjBlock(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [
                nn.Linear(embedding_dim, embedding_dim),
                nn.LeakyReLU(0.3),
                nn.Linear(embedding_dim, embedding_dim),
                nn.LeakyReLU(0.3),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        for layer in self.layers:
            x = layer(x)
        x = x + res
        return x


class LinearNorm(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias)

        nn.init.xavier_uniform_(self.linear.weight)
        if bias:
            nn.init.constant_(self.linear.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return x


class STL(nn.Module):
    """
    A PyTorch module for the Style Token Layer (STL) as described in
    "A Style-Based Generator Architecture for Generative Adversarial Networks"
    (https://arxiv.org/abs/1812.04948)

    The STL applies a multi-headed attention mechanism over the learned style tokens,
    using the text input as the query and the style tokens as the keys and values.
    The output of the attention mechanism is used as the text's style embedding.

    Args:
        token_num (int): The number of style tokens.
        n_hidden (int): Number of hidden dimensions.
    """

    def __init__(self, n_hidden: int, token_num: int):
        super(STL, self).__init__()  # pylint: disable=super-with-arguments

        num_heads = 1
        E = n_hidden
        self.token_num = token_num
        self.embed = nn.Parameter(torch.FloatTensor(self.token_num, E // num_heads))
        d_q = E // 2
        d_k = E // num_heads
        self.attention = StyleEmbedAttention(query_dim=d_q, key_dim=d_k, num_units=E, num_heads=num_heads)

        torch.nn.init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        query = x.unsqueeze(1)  # [N, 1, E//2]

        keys_soft = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, E // num_heads]

        # Weighted sum
        emotion_embed_soft = self.attention(query, keys_soft)

        return emotion_embed_soft
