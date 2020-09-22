import math
import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.glow import LayerNorm


class RelativePositionMultiHeadAttention(nn.Module):
    """Implementation of Relative Position Encoding based on
    https://arxiv.org/pdf/1809.04281.pdf
    """
    def __init__(self,
                 channels,
                 out_channels,
                 num_heads,
                 rel_attn_window_size=None,
                 heads_share=True,
                 dropout_p=0.,
                 input_length=None,
                 proximal_bias=False,
                 proximal_init=False):
        super().__init__()
        assert channels % num_heads == 0, " [!] channels should be divisible by num_heads."
        # class attributes
        self.channels = channels
        self.out_channels = out_channels
        self.num_heads = num_heads
        self.rel_attn_window_size = rel_attn_window_size
        self.heads_share = heads_share
        self.input_length = input_length
        self.proximal_bias = proximal_bias
        self.dropout_p = dropout_p
        self.attn = None
        # query, key, value layers
        self.k_channels = channels // num_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        # output layers
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.dropout = nn.Dropout(dropout_p)
        # relative positional encoding layers
        if rel_attn_window_size is not None:
            n_heads_rel = 1 if heads_share else num_heads
            rel_stddev = self.k_channels**-0.5
            emb_rel_k = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1,
                            self.k_channels) * rel_stddev)
            emb_rel_v = nn.Parameter(
                torch.randn(n_heads_rel, rel_attn_window_size * 2 + 1,
                            self.k_channels) * rel_stddev)
            self.register_parameter('emb_rel_k', emb_rel_k)
            self.register_parameter('emb_rel_v', emb_rel_v)

        # init layers
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        # proximal bias
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, c, attn_mask=None):
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x

    def attention(self, query, key, value, mask=None):
        # reshape [b, d, t] -> [b, n_h, t, d_k]
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = query.view(b, self.num_heads, self.k_channels,
                           t_t).transpose(2, 3)
        key = key.view(b, self.num_heads, self.k_channels, t_s).transpose(2, 3)
        value = value.view(b, self.num_heads, self.k_channels,
                           t_s).transpose(2, 3)
        # compute raw attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            self.k_channels)
        # relative positional encoding
        if self.rel_attn_window_size is not None:
            assert t_s == t_t, "Relative attention is only available for self-attention."
            # get relative key embeddings
            key_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_k, t_s)
            rel_logits = self._matmul_with_relative_keys(
                query, key_relative_embeddings)
            rel_logits = self._relative_position_to_absolute_position(
                rel_logits)
            scores_local = rel_logits / math.sqrt(self.k_channels)
            scores = scores + scores_local
        # proximan bias
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attn_proximity_bias(t_s).to(
                device=scores.device, dtype=scores.dtype)
        # attention score masking
        if mask is not None:
            # add small value to prevent oor error.
            scores = scores.masked_fill(mask == 0, -1e4)
            if self.input_length is not None:
                block_mask = torch.ones_like(scores).triu(
                    -1 * self.input_length).tril(self.input_length)
                scores = scores * block_mask + -1e4 * (1 - block_mask)
        # attention score normalization
        p_attn = F.softmax(scores, dim=-1)  # [b, n_h, t_t, t_s]
        # apply dropout to attention weights
        p_attn = self.dropout(p_attn)
        # compute output
        output = torch.matmul(p_attn, value)
        # relative positional encoding for values
        if self.rel_attn_window_size is not None:
            relative_weights = self._absolute_position_to_relative_position(
                p_attn)
            value_relative_embeddings = self._get_relative_embeddings(
                self.emb_rel_v, t_s)
            output = output + self._matmul_with_relative_values(
                relative_weights, value_relative_embeddings)
        output = output.transpose(2, 3).contiguous().view(
            b, d, t_t)  # [b, n_h, t_t, d_k] -> [b, d, t_t]
        return output, p_attn

    @staticmethod
    def _matmul_with_relative_values(p_attn, re):
        """
        Args:
            p_attn (Tensor): attention weights.
            re (Tensor): relative value embedding vector. (a_(i,j)^V)

        Shapes:
            p_attn: [B, H, T, V]
            re: [H or 1, V, D]
            logits: [B, H, T, D]
        """
        logits = torch.matmul(p_attn, re.unsqueeze(0))
        return logits

    @staticmethod
    def _matmul_with_relative_keys(query, re):
        """
        Args:
            query (Tensor): batch of query vectors. (x*W^Q)
            re (Tensor): relative key embedding vector. (a_(i,j)^K)

        Shapes:
            query: [B, H, T, D]
            re: [H or 1, V, D]
            logits: [B, H, T, V]
        """
        # logits = torch.einsum('bhld, kmd -> bhlm', [query, re.to(query.dtype)])
        logits = torch.matmul(query, re.unsqueeze(0).transpose(-2, -1))
        return logits

    def _get_relative_embeddings(self, relative_embeddings, length):
        """Convert embedding vestors to a tensor of embeddings
        """
        # Pad first before slice to avoid using cond ops.
        pad_length = max(length - (self.rel_attn_window_size + 1), 0)
        slice_start_position = max((self.rel_attn_window_size + 1) - length, 0)
        slice_end_position = slice_start_position + 2 * length - 1
        if pad_length > 0:
            padded_relative_embeddings = F.pad(
                relative_embeddings, [0, 0, pad_length, pad_length, 0, 0])
        else:
            padded_relative_embeddings = relative_embeddings
        used_relative_embeddings = padded_relative_embeddings[:,
                                                              slice_start_position:
                                                              slice_end_position]
        return used_relative_embeddings

    @staticmethod
    def _relative_position_to_absolute_position(x):
        """Converts tensor from relative to absolute indexing for local attention.
        Args:
            x: [B, D, length, 2 * length - 1]
        Returns:
            A Tensor of shape [B, D, length, length]
        """
        batch, heads, length, _ = x.size()
        # Pad to shift from relative to absolute indexing.
        x = F.pad(x, [0, 1, 0, 0, 0, 0, 0, 0])
        # Pad extra elements so to add up to shape (len+1, 2*len-1).
        x_flat = x.view([batch, heads, length * 2 * length])
        x_flat = F.pad(x_flat, [0, length - 1, 0, 0, 0, 0])
        # Reshape and slice out the padded elements.
        x_final = x_flat.view([batch, heads, length + 1,
                               2 * length - 1])[:, :, :length, length - 1:]
        return x_final

    @staticmethod
    def _absolute_position_to_relative_position(x):
        """
        x: [B, H, T, T]
        ret: [B, H, T, 2*T-1]
        """
        batch, heads, length, _ = x.size()
        # padd along column
        x = F.pad(x, [0, length - 1, 0, 0, 0, 0, 0, 0])
        x_flat = x.view([batch, heads, length**2 + length * (length - 1)])
        # add 0's in the beginning that will skew the elements after reshape
        x_flat = F.pad(x_flat, [length, 0, 0, 0, 0, 0])
        x_final = x_flat.view([batch, heads, length, 2 * length])[:, :, :, 1:]
        return x_final

    @staticmethod
    def _attn_proximity_bias(length):
        """Produce an attention mask that discourages distant
        attention values.
        Args:
            length (int): an integer scalar.
        Returns:
            a Tensor with shape [1, 1, length, length]
        """
        # L
        r = torch.arange(length, dtype=torch.float32)
        # L x L
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        # scale mask values
        diff = -torch.log1p(torch.abs(diff))
        # 1 x 1 x L x L
        return diff.unsqueeze(0).unsqueeze(0)


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
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x, x_mask):
        x = self.conv_1(x * x_mask)
        if self.activation == "gelu":
            x = x * torch.sigmoid(1.702 * x)
        else:
            x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask


class Transformer(nn.Module):
    def __init__(self,
                 hidden_channels,
                 filter_channels,
                 num_heads,
                 num_layers,
                 kernel_size=1,
                 dropout_p=0.,
                 rel_attn_window_size=None,
                 input_length=None):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.filter_channels = filter_channels
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.kernel_size = kernel_size
        self.dropout_p = dropout_p
        self.rel_attn_window_size = rel_attn_window_size

        self.dropout = nn.Dropout(dropout_p)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(self.num_layers):
            self.attn_layers.append(
                RelativePositionMultiHeadAttention(
                    hidden_channels,
                    hidden_channels,
                    num_heads,
                    rel_attn_window_size=rel_attn_window_size,
                    dropout_p=dropout_p,
                    input_length=input_length))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(
                FFN(hidden_channels,
                    hidden_channels,
                    filter_channels,
                    kernel_size,
                    dropout_p=dropout_p))
            self.norm_layers_2.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(self.num_layers):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.dropout(y)
            x = self.norm_layers_1[i](x + y)

            y = self.ffn_layers[i](x, x_mask)
            y = self.dropout(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x
