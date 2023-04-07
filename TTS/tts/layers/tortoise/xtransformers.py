import math
from collections import namedtuple
from functools import partial
from inspect import isfunction

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import nn, einsum

DEFAULT_DIM_HEAD = 64

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates',
    'past_key_values',
])


# helpers

def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def cast_tuple(val, depth):
    return val if isinstance(val, tuple) else (val,) * depth


class always():
    def __init__(self, val):
        self.val = val

    def __call__(self, *args, **kwargs):
        return self.val


class not_equals():
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x != self.val


class equals():
    def __init__(self, val):
        self.val = val

    def __call__(self, x, *args, **kwargs):
        return x == self.val


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def l2norm(t):
    return F.normalize(t, p=2, dim=-1)


# init helpers

def init_zero_(layer):
    nn.init.constant_(layer.weight, 0.)
    if exists(layer.bias):
        nn.init.constant_(layer.bias, 0.)


# keyword argument helpers

def pick_and_pop(keys, d):
    values = list(map(lambda key: d.pop(key), keys))
    return dict(zip(keys, values))


def group_dict_by_key(cond, d):
    return_val = [dict(), dict()]
    for key in d.keys():
        match = bool(cond(key))
        ind = int(not match)
        return_val[ind][key] = d[key]
    return (*return_val,)


def string_begins_with(prefix, str):
    return str.startswith(prefix)


def group_by_key_prefix(prefix, d):
    return group_dict_by_key(partial(string_begins_with, prefix), d)


def groupby_prefix_and_trim(prefix, d):
    kwargs_with_prefix, kwargs = group_dict_by_key(partial(string_begins_with, prefix), d)
    kwargs_without_prefix = dict(map(lambda x: (x[0][len(prefix):], x[1]), tuple(kwargs_with_prefix.items())))
    return kwargs_without_prefix, kwargs


# activations

class ReluSquared(nn.Module):
    def forward(self, x):
        return F.relu(x) ** 2


# positional embeddings

class AbsolutePositionalEmbedding(nn.Module):
    def __init__(self, dim, max_seq_len):
        super().__init__()
        self.scale = dim ** -0.5
        self.emb = nn.Embedding(max_seq_len, dim)

    def forward(self, x):
        n = torch.arange(x.shape[1], device=x.device)
        pos_emb = self.emb(n)
        pos_emb = rearrange(pos_emb, 'n d -> () n d')
        return pos_emb * self.scale


class FixedPositionalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x, seq_dim=1, offset=0):
        t = torch.arange(x.shape[seq_dim], device=x.device).type_as(self.inv_freq) + offset
        sinusoid_inp = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((sinusoid_inp.sin(), sinusoid_inp.cos()), dim=-1)
        return rearrange(emb, 'n d -> () n d')


class RelativePositionBias(nn.Module):
    def __init__(self, scale, causal=False, num_buckets=32, max_distance=128, heads=8):
        super().__init__()
        self.scale = scale
        self.causal = causal
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, causal=True, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position
        if not causal:
            num_buckets //= 2
            ret += (n < 0).long() * num_buckets
            n = torch.abs(n)
        else:
            n = torch.max(n, torch.zeros_like(n))

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = max_exact + (
                torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)
        ).long()
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, qk_dots):
        i, j, device = *qk_dots.shape[-2:], qk_dots.device
        q_pos = torch.arange(i, dtype=torch.long, device=device)
        k_pos = torch.arange(j, dtype=torch.long, device=device)
        rel_pos = k_pos[None, :] - q_pos[:, None]
        rp_bucket = self._relative_position_bucket(rel_pos, causal=self.causal, num_buckets=self.num_buckets,
                                                   max_distance=self.max_distance)
        values = self.relative_attention_bias(rp_bucket)
        bias = rearrange(values, 'i j h -> () h i j')
        return qk_dots + (bias * self.scale)


class AlibiPositionalBias(nn.Module):
    def __init__(self, heads, **kwargs):
        super().__init__()
        self.heads = heads
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> () h () ()')
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    @staticmethod
    def _get_slopes(heads):
        def get_slopes_power_of_2(n):
            start = (2 ** (-2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * ratio ** i for i in range(n)]

        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return get_slopes_power_of_2(closest_power_of_2) + get_slopes_power_of_2(2 * closest_power_of_2)[0::2][
                                                           :heads - closest_power_of_2]

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        if exists(self.bias) and self.bias.shape[-1] >= j:
            return qk_dots + self.bias[..., :j]

        bias = torch.arange(j, device=device)
        bias = rearrange(bias, 'j -> () () () j')
        bias = bias * self.slopes

        num_heads_unalibied = h - bias.shape[1]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))

        self.register_buffer('bias', bias, persistent=False)
        return qk_dots + self.bias


class LearnedAlibiPositionalBias(AlibiPositionalBias):
    def __init__(self, heads, bidirectional=False):
        super().__init__(heads)
        los_slopes = torch.log(self.slopes)
        self.learned_logslopes = nn.Parameter(los_slopes)

        self.bidirectional = bidirectional
        if self.bidirectional:
            self.learned_logslopes_future = nn.Parameter(los_slopes)

    def forward(self, qk_dots):
        h, i, j, device = *qk_dots.shape[-3:], qk_dots.device

        def get_slopes(param):
            return F.pad(param.exp(), (0, 0, 0, 0, 0, h - param.shape[1]))

        if exists(self.bias) and self.bias.shape[-1] >= j:
            bias = self.bias[..., :i, :j]
        else:
            i_arange = torch.arange(i, device=device)
            j_arange = torch.arange(j, device=device)
            bias = rearrange(j_arange, 'j -> 1 1 1 j') - rearrange(i_arange, 'i -> 1 1 i 1')
            self.register_buffer('bias', bias, persistent=False)

        if self.bidirectional:
            past_slopes = get_slopes(self.learned_logslopes)
            future_slopes = get_slopes(self.learned_logslopes_future)
            bias = torch.tril(bias * past_slopes) + torch.triu(bias * future_slopes)
        else:
            slopes = get_slopes(self.learned_logslopes)
            bias = bias * slopes

        return qk_dots + bias


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, device):
        t = torch.arange(max_seq_len, device=device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> () () n d')


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len = t.shape[-2]
    freqs = freqs[:, :, -seq_len:]
    return (t * freqs.cos()) + (rotate_half(t) * freqs.sin())


# norms

class Scale(nn.Module):
    def __init__(self, value, fn):
        super().__init__()
        self.value = value
        self.fn = fn

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        scale_fn = lambda t: t * self.value

        if not isinstance(out, tuple):
            return scale_fn(out)

        return (scale_fn(out[0]), *out[1:])


class Rezero(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.g = nn.Parameter(torch.zeros(1))

    def forward(self, x, **kwargs):
        out = self.fn(x, **kwargs)
        rezero_fn = lambda t: t * self.g

        if not isinstance(out, tuple):
            return rezero_fn(out)

        return (rezero_fn(out[0]), *out[1:])


class ScaleNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g


class RMSScaleShiftNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))
        self.scale_shift_process = nn.Linear(dim * 2, dim * 2)

    def forward(self, x, norm_scale_shift_inp):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        norm = x / norm.clamp(min=self.eps) * self.g

        ss_emb = self.scale_shift_process(norm_scale_shift_inp)
        scale, shift = torch.chunk(ss_emb, 2, dim=1)
        h = norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return h


# residual and residual gates

class Residual(nn.Module):
    def __init__(self, dim, scale_residual=False):
        super().__init__()
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        return x + residual


class GRUGating(nn.Module):
    def __init__(self, dim, scale_residual=False):
        super().__init__()
        self.gru = nn.GRUCell(dim, dim)
        self.residual_scale = nn.Parameter(torch.ones(dim)) if scale_residual else None

    def forward(self, x, residual):
        if exists(self.residual_scale):
            residual = residual * self.residual_scale

        gated_output = self.gru(
            rearrange(x, 'b n d -> (b n) d'),
            rearrange(residual, 'b n d -> (b n) d')
        )

        return gated_output.reshape_as(x)


# token shifting

def shift(t, amount, mask=None):
    if amount == 0:
        return t

    if exists(mask):
        t = t.masked_fill(~mask[..., None], 0.)

    return F.pad(t, (0, 0, amount, -amount), value=0.)


class ShiftTokens(nn.Module):
    def __init__(self, shifts, fn):
        super().__init__()
        self.fn = fn
        self.shifts = tuple(shifts)

    def forward(self, x, **kwargs):
        mask = kwargs.get('mask', None)
        shifts = self.shifts
        segments = len(shifts)
        feats_per_shift = x.shape[-1] // segments
        splitted = x.split(feats_per_shift, dim=-1)
        segments_to_shift, rest = splitted[:segments], splitted[segments:]
        segments_to_shift = list(map(lambda args: shift(*args, mask=mask), zip(segments_to_shift, shifts)))
        x = torch.cat((*segments_to_shift, *rest), dim=-1)
        return self.fn(x, **kwargs)


# feedforward

class GLU(nn.Module):
    def __init__(self, dim_in, dim_out, activation):
        super().__init__()
        self.act = activation
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * self.act(gate)


class FeedForward(nn.Module):
    def __init__(
            self,
            dim,
            dim_out=None,
            mult=4,
            glu=False,
            relu_squared=False,
            post_act_ln=False,
            dropout=0.,
            zero_init_output=False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        activation = ReluSquared() if relu_squared else nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            activation
        ) if not glu else GLU(dim, inner_dim, activation)

        self.net = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

        # init last linear layer to 0
        if zero_init_output:
            init_zero_(self.net[-1])

    def forward(self, x):
        return self.net(x)


# attention.

class Attention(nn.Module):
    def __init__(
            self,
            dim,
            dim_head=DEFAULT_DIM_HEAD,
            heads=8,
            causal=False,
            talking_heads=False,
            head_scale=False,
            collab_heads=False,
            collab_compression=.3,
            sparse_topk=None,
            use_entmax15=False,
            num_mem_kv=0,
            dropout=0.,
            on_attn=False,
            gate_values=False,
            zero_init_output=False,
            max_attend_past=None,
            qk_norm=False,
            scale_init_value=None,
            rel_pos_bias=False,
            rel_pos_num_buckets=32,
            rel_pos_max_distance=128,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        self.heads = heads
        self.causal = causal
        self.max_attend_past = max_attend_past

        qk_dim = v_dim = dim_head * heads

        # collaborative heads
        self.collab_heads = collab_heads
        if self.collab_heads:
            qk_dim = int(collab_compression * qk_dim)
            self.collab_mixing = nn.Parameter(torch.randn(heads, qk_dim))

        self.to_q = nn.Linear(dim, qk_dim, bias=False)
        self.to_k = nn.Linear(dim, qk_dim, bias=False)
        self.to_v = nn.Linear(dim, v_dim, bias=False)

        self.dropout = nn.Dropout(dropout)

        # add GLU gating for aggregated values, from alphafold2
        self.to_v_gate = None
        if gate_values:
            self.to_v_gate = nn.Linear(dim, v_dim)
            nn.init.constant_(self.to_v_gate.weight, 0)
            nn.init.constant_(self.to_v_gate.bias, 1)

        # cosine sim attention
        self.qk_norm = qk_norm
        if qk_norm:
            scale_init_value = default(scale_init_value,
                                       -3)  # if not provided, initialize as though it were sequence length of 1024
            self.scale = nn.Parameter(torch.ones(1, heads, 1, 1) * scale_init_value)

        # talking heads
        self.talking_heads = talking_heads
        if talking_heads:
            self.pre_softmax_proj = nn.Parameter(torch.randn(heads, heads))
            self.post_softmax_proj = nn.Parameter(torch.randn(heads, heads))

        # head scaling
        self.head_scale = head_scale
        if head_scale:
            self.head_scale_params = nn.Parameter(torch.ones(1, heads, 1, 1))

        # explicit topk sparse attention
        self.sparse_topk = sparse_topk

        # entmax
        self.attn_fn = F.softmax

        # add memory key / values
        self.num_mem_kv = num_mem_kv
        if num_mem_kv > 0:
            self.mem_k = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))
            self.mem_v = nn.Parameter(torch.randn(heads, num_mem_kv, dim_head))

        # attention on attention
        self.attn_on_attn = on_attn
        self.to_out = nn.Sequential(nn.Linear(v_dim, dim * 2), nn.GLU()) if on_attn else nn.Linear(v_dim, dim)

        self.rel_pos_bias = rel_pos_bias
        if rel_pos_bias:
            assert rel_pos_num_buckets <= rel_pos_max_distance, 'number of relative position buckets must be less than the relative position max distance'
            self.rel_pos = RelativePositionBias(scale=dim_head ** 0.5, causal=causal, heads=heads,
                                                num_buckets=rel_pos_num_buckets, max_distance=rel_pos_max_distance)

        # init output projection 0
        if zero_init_output:
            init_zero_(self.to_out)

    def forward(
            self,
            x,
            context=None,
            mask=None,
            context_mask=None,
            attn_mask=None,
            sinusoidal_emb=None,
            rotary_pos_emb=None,
            prev_attn=None,
            mem=None,
            layer_past=None,
    ):
        b, n, _, h, talking_heads, collab_heads, head_scale, scale, device, has_context = *x.shape, self.heads, self.talking_heads, self.collab_heads, self.head_scale, self.scale, x.device, exists(
            context)
        kv_input = default(context, x)

        q_input = x
        k_input = kv_input
        v_input = kv_input

        if exists(mem):
            k_input = torch.cat((mem, k_input), dim=-2)
            v_input = torch.cat((mem, v_input), dim=-2)

        if exists(sinusoidal_emb):
            # in shortformer, the query would start at a position offset depending on the past cached memory
            offset = k_input.shape[-2] - q_input.shape[-2]
            q_input = q_input + sinusoidal_emb(q_input, offset=offset)
            k_input = k_input + sinusoidal_emb(k_input)

        q = self.to_q(q_input)
        k = self.to_k(k_input)
        v = self.to_v(v_input)

        if not collab_heads:
            q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        else:
            q = einsum('b i d, h d -> b h i d', q, self.collab_mixing)
            k = rearrange(k, 'b n d -> b () n d')
            v = rearrange(v, 'b n (h d) -> b h n d', h=h)

        if layer_past is not None:
            past_key, past_value = layer_past
            k = torch.cat([past_key, k], dim=-2)
            v = torch.cat([past_value, v], dim=-2)
        k_cache = k
        v_cache = v

        if exists(rotary_pos_emb) and not has_context:
            l = rotary_pos_emb.shape[-1]
            (ql, qr), (kl, kr), (vl, vr) = map(lambda t: (t[..., :l], t[..., l:]), (q, k, v))
            ql, kl, vl = map(lambda t: apply_rotary_pos_emb(t, rotary_pos_emb), (ql, kl, vl))
            q, k, v = map(lambda t: torch.cat(t, dim=-1), ((ql, qr), (kl, kr), (vl, vr)))

        input_mask = None
        if any(map(exists, (mask, context_mask))):
            q_mask = default(mask, lambda: torch.ones((b, n), device=device).bool())
            k_mask = q_mask if not exists(context) else context_mask
            k_mask = default(k_mask, lambda: torch.ones((b, k.shape[-2]), device=device).bool())
            q_mask = rearrange(q_mask, 'b i -> b () i ()')
            k_mask = rearrange(k_mask, 'b j -> b () () j')
            input_mask = q_mask * k_mask

        if self.num_mem_kv > 0:
            mem_k, mem_v = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), (self.mem_k, self.mem_v))
            k = torch.cat((mem_k, k), dim=-2)
            v = torch.cat((mem_v, v), dim=-2)
            if exists(input_mask):
                input_mask = F.pad(input_mask, (self.num_mem_kv, 0), value=True)

        if collab_heads:
            k = k.expand(-1, h, -1, -1)

        if self.qk_norm:
            q, k = map(l2norm, (q, k))
            scale = 1 / (self.scale.exp().clamp(min=1e-2))

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * scale
        mask_value = max_neg_value(dots)

        if exists(prev_attn):
            dots = dots + prev_attn

        pre_softmax_attn = dots.clone()

        if talking_heads:
            dots = einsum('b h i j, h k -> b k i j', dots, self.pre_softmax_proj).contiguous()

        if self.rel_pos_bias:
            dots = self.rel_pos(dots)

        if exists(input_mask):
            dots.masked_fill_(~input_mask, mask_value)
            del input_mask

        if exists(attn_mask):
            assert 2 <= attn_mask.ndim <= 4, 'attention mask must have greater than 2 dimensions but less than or equal to 4'
            if attn_mask.ndim == 2:
                attn_mask = rearrange(attn_mask, 'i j -> () () i j')
            elif attn_mask.ndim == 3:
                attn_mask = rearrange(attn_mask, 'h i j -> () h i j')
            dots.masked_fill_(~attn_mask, mask_value)

        if exists(self.max_attend_past):
            i, j = dots.shape[-2:]
            range_q = torch.arange(j - i, j, device=device)
            range_k = torch.arange(j, device=device)
            dist = rearrange(range_q, 'i -> () () i ()') - rearrange(range_k, 'j -> () () () j')
            mask = dist > self.max_attend_past
            dots.masked_fill_(mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            r = torch.arange(i, device=device)
            mask = rearrange(r, 'i -> () () i ()') < rearrange(r, 'j -> () () () j')
            mask = F.pad(mask, (j - i, 0), value=False)
            dots.masked_fill_(mask, mask_value)
            del mask

        if exists(self.sparse_topk) and self.sparse_topk < dots.shape[-1]:
            top, _ = dots.topk(self.sparse_topk, dim=-1)
            vk = top[..., -1].unsqueeze(-1).expand_as(dots)
            mask = dots < vk
            dots.masked_fill_(mask, mask_value)
            del mask

        attn = self.attn_fn(dots, dim=-1)
        post_softmax_attn = attn.clone()

        attn = self.dropout(attn)

        if talking_heads:
            attn = einsum('b h i j, h k -> b k i j', attn, self.post_softmax_proj).contiguous()

        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        if head_scale:
            out = out * self.head_scale_params

        out = rearrange(out, 'b h n d -> b n (h d)')

        if exists(self.to_v_gate):
            gates = self.to_v_gate(x)
            out = out * gates.sigmoid()

        intermediates = Intermediates(
            pre_softmax_attn=pre_softmax_attn,
            post_softmax_attn=post_softmax_attn
        )

        return self.to_out(out), intermediates, k_cache, v_cache


class AttentionLayers(nn.Module):
    def __init__(
            self,
            dim,
            depth,
            heads=8,
            causal=False,
            cross_attend=False,
            only_cross=False,
            use_scalenorm=False,
            use_rms_scaleshift_norm=False,
            use_rmsnorm=False,
            use_rezero=False,
            alibi_pos_bias=False,
            alibi_num_heads=None,
            alibi_learned=False,
            position_infused_attn=False,
            rotary_pos_emb=False,
            rotary_emb_dim=None,
            custom_layers=None,
            sandwich_coef=None,
            par_ratio=None,
            residual_attn=False,
            cross_residual_attn=False,
            macaron=False,
            pre_norm=True,
            gate_residual=False,
            scale_residual=False,
            shift_tokens=0,
            sandwich_norm=False,
            use_qk_norm_attn=False,
            qk_norm_attn_seq_len=None,
            zero_init_branch_output=False,
            **kwargs
    ):
        super().__init__()
        ff_kwargs, kwargs = groupby_prefix_and_trim('ff_', kwargs)
        attn_kwargs, _ = groupby_prefix_and_trim('attn_', kwargs)

        dim_head = attn_kwargs.get('dim_head', DEFAULT_DIM_HEAD)

        self.dim = dim
        self.depth = depth
        self.layers = nn.ModuleList([])
        self.causal = causal

        rel_pos_bias = 'rel_pos_bias' in attn_kwargs
        self.has_pos_emb = position_infused_attn or rel_pos_bias or rotary_pos_emb
        self.pia_pos_emb = FixedPositionalEmbedding(dim) if position_infused_attn else None

        rotary_emb_dim = max(default(rotary_emb_dim, dim_head // 2), 32)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim) if rotary_pos_emb else None

        assert not (
                    alibi_pos_bias and rel_pos_bias), 'you can only choose Alibi positional bias or T5 relative positional bias, not both'

        if alibi_pos_bias:
            alibi_num_heads = default(alibi_num_heads, heads)
            assert alibi_num_heads <= heads, 'number of ALiBi heads must be less than the total number of heads'
            alibi_pos_klass = LearnedAlibiPositionalBias if alibi_learned or not causal else AlibiPositionalBias
            self.rel_pos = alibi_pos_klass(heads=alibi_num_heads, bidirectional=not causal)
        else:
            self.rel_pos = None

        assert not (not pre_norm and sandwich_norm), 'sandwich norm cannot be used when not using prenorm'
        self.pre_norm = pre_norm
        self.sandwich_norm = sandwich_norm

        self.residual_attn = residual_attn
        self.cross_residual_attn = cross_residual_attn
        self.cross_attend = cross_attend

        norm_class = ScaleNorm if use_scalenorm else nn.LayerNorm
        norm_class = RMSNorm if use_rmsnorm else norm_class
        norm_class = RMSScaleShiftNorm if use_rms_scaleshift_norm else norm_class
        norm_fn = partial(norm_class, dim)

        norm_fn = nn.Identity if use_rezero else norm_fn
        branch_fn = Rezero if use_rezero else None

        if cross_attend and not only_cross:
            default_block = ('a', 'c', 'f')
        elif cross_attend and only_cross:
            default_block = ('c', 'f')
        else:
            default_block = ('a', 'f')

        if macaron:
            default_block = ('f',) + default_block

        # qk normalization

        if use_qk_norm_attn:
            attn_scale_init_value = -math.log(math.log2(qk_norm_attn_seq_len ** 2 - qk_norm_attn_seq_len)) if exists(
                qk_norm_attn_seq_len) else None
            attn_kwargs = {**attn_kwargs, 'qk_norm': True, 'scale_init_value': attn_scale_init_value}

        # zero init

        if zero_init_branch_output:
            attn_kwargs = {**attn_kwargs, 'zero_init_output': True}
            ff_kwargs = {**ff_kwargs, 'zero_init_output': True}

        # calculate layer block order

        if exists(custom_layers):
            layer_types = custom_layers
        elif exists(par_ratio):
            par_depth = depth * len(default_block)
            assert 1 < par_ratio <= par_depth, 'par ratio out of range'
            default_block = tuple(filter(not_equals('f'), default_block))
            par_attn = par_depth // par_ratio
            depth_cut = par_depth * 2 // 3  # 2 / 3 attention layer cutoff suggested by PAR paper
            par_width = (depth_cut + depth_cut // par_attn) // par_attn
            assert len(default_block) <= par_width, 'default block is too large for par_ratio'
            par_block = default_block + ('f',) * (par_width - len(default_block))
            par_head = par_block * par_attn
            layer_types = par_head + ('f',) * (par_depth - len(par_head))
        elif exists(sandwich_coef):
            assert sandwich_coef > 0 and sandwich_coef <= depth, 'sandwich coefficient should be less than the depth'
            layer_types = ('a',) * sandwich_coef + default_block * (depth - sandwich_coef) + ('f',) * sandwich_coef
        else:
            layer_types = default_block * depth

        self.layer_types = layer_types
        self.num_attn_layers = len(list(filter(equals('a'), layer_types)))

        # calculate token shifting

        shift_tokens = cast_tuple(shift_tokens, len(layer_types))

        # iterate and construct layers

        for ind, (layer_type, layer_shift_tokens) in enumerate(zip(self.layer_types, shift_tokens)):
            is_last_layer = ind == (len(self.layer_types) - 1)

            if layer_type == 'a':
                layer = Attention(dim, heads=heads, causal=causal, **attn_kwargs)
            elif layer_type == 'c':
                layer = Attention(dim, heads=heads, **attn_kwargs)
            elif layer_type == 'f':
                layer = FeedForward(dim, **ff_kwargs)
                layer = layer if not macaron else Scale(0.5, layer)
            else:
                raise Exception(f'invalid layer type {layer_type}')

            if layer_shift_tokens > 0:
                shift_range_upper = layer_shift_tokens + 1
                shift_range_lower = -layer_shift_tokens if not causal else 0
                layer = ShiftTokens(range(shift_range_lower, shift_range_upper), layer)

            if exists(branch_fn):
                layer = branch_fn(layer)

            residual_fn = GRUGating if gate_residual else Residual
            residual = residual_fn(dim, scale_residual=scale_residual)

            layer_uses_qk_norm = use_qk_norm_attn and layer_type in ('a', 'c')

            pre_branch_norm = norm_fn() if pre_norm and not layer_uses_qk_norm else None
            post_branch_norm = norm_fn() if sandwich_norm or layer_uses_qk_norm else None
            post_main_norm = norm_fn() if not pre_norm and not is_last_layer else None

            norms = nn.ModuleList([
                pre_branch_norm,
                post_branch_norm,
                post_main_norm
            ])

            self.layers.append(nn.ModuleList([
                norms,
                layer,
                residual
            ]))

    def forward(
            self,
            x,
            context=None,
            full_context=None,  # for passing a list of hidden states from an encoder
            mask=None,
            context_mask=None,
            attn_mask=None,
            mems=None,
            return_hiddens=False,
            norm_scale_shift_inp=None,
            past_key_values=None,
            expected_seq_len=None,
    ):

        assert not (self.cross_attend ^ (exists(context) or exists(
            full_context))), 'context must be passed in if cross_attend is set to True'
        assert context is None or full_context is None, 'only one of full_context or context can be provided'

        hiddens = []
        intermediates = []
        prev_attn = None
        prev_cross_attn = None

        mems = mems.copy() if exists(mems) else [None] * self.num_attn_layers
        norm_args = {}
        if exists(norm_scale_shift_inp):
            norm_args['norm_scale_shift_inp'] = norm_scale_shift_inp

        rotary_pos_emb = None
        if exists(self.rotary_pos_emb):
            if not self.training and self.causal:
                assert expected_seq_len is not None, "To decode a transformer with rotary embeddings, you must specify an `expected_seq_len`"
            elif expected_seq_len is None:
                expected_seq_len = 0
            seq_len = x.shape[1]
            if past_key_values is not None:
                seq_len += past_key_values[0][0].shape[-2]
            max_rotary_emb_length = max(list(map(lambda m: (m.shape[1] if exists(m) else 0) + seq_len, mems)) + [expected_seq_len])
            rotary_pos_emb = self.rotary_pos_emb(max_rotary_emb_length, x.device)

        present_key_values = []
        cross_attn_count = 0
        for ind, (layer_type, (norm, block, residual_fn)) in enumerate(zip(self.layer_types, self.layers)):
            if layer_type == 'a':
                layer_mem = mems.pop(0) if mems else None

            residual = x

            pre_branch_norm, post_branch_norm, post_main_norm = norm

            if exists(pre_branch_norm):
                x = pre_branch_norm(x, **norm_args)

            if layer_type == 'a' or layer_type == 'c':
                if past_key_values is not None:
                    layer_kv = past_key_values.pop(0)
                    layer_past = tuple(s.to(x.device) for s in layer_kv)
                else:
                    layer_past = None

            if layer_type == 'a':
                out, inter, k, v = block(x, None, mask, None, attn_mask, self.pia_pos_emb, rotary_pos_emb,
                                        prev_attn, layer_mem, layer_past)
            elif layer_type == 'c':
                if exists(full_context):
                    out, inter, k, v = block(x, full_context[cross_attn_count], mask, context_mask, None, None,
                                            None, prev_attn, None, layer_past)
                else:
                    out, inter, k, v = block(x, context, mask, context_mask, None, None, None, prev_attn, None, layer_past)
            elif layer_type == 'f':
                out = block(x)

            if layer_type == 'a' or layer_type == 'c' and present_key_values is not None:
                present_key_values.append((k.detach(), v.detach()))

            if exists(post_branch_norm):
                out = post_branch_norm(out, **norm_args)

            x = residual_fn(out, residual)

            if layer_type in ('a', 'c'):
                intermediates.append(inter)

            if layer_type == 'a' and self.residual_attn:
                prev_attn = inter.pre_softmax_attn
            elif layer_type == 'c' and self.cross_residual_attn:
                prev_cross_attn = inter.pre_softmax_attn

            if exists(post_main_norm):
                x = post_main_norm(x, **norm_args)

            if layer_type == 'c':
                cross_attn_count += 1

            if layer_type == 'f':
                hiddens.append(x)

        if return_hiddens:
            intermediates = LayerIntermediates(
                hiddens=hiddens,
                attn_intermediates=intermediates,
                past_key_values=present_key_values
            )

            return x, intermediates

        return x


class Encoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on encoder'
        super().__init__(causal=False, **kwargs)


class Decoder(AttentionLayers):
    def __init__(self, **kwargs):
        assert 'causal' not in kwargs, 'cannot set causality on decoder'
        super().__init__(causal=True, **kwargs)


class CrossAttender(AttentionLayers):
    def __init__(self, **kwargs):
        super().__init__(cross_attend=True, only_cross=True, **kwargs)


class ViTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            image_size,
            patch_size,
            attn_layers,
            num_classes=None,
            dropout=0.,
            emb_dropout=0.
    ):
        super().__init__()
        assert isinstance(attn_layers, Encoder), 'attention layers must be an Encoder'
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        dim = attn_layers.dim
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 3 * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = FeedForward(dim, dim_out=num_classes, dropout=dropout) if exists(num_classes) else None

    def forward(
            self,
            img,
            return_embeddings=False
    ):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.attn_layers(x)
        x = self.norm(x)

        if not exists(self.mlp_head) or return_embeddings:
            return x

        return self.mlp_head(x[:, 0])


class TransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            num_tokens,
            max_seq_len,
            attn_layers,
            emb_dim=None,
            max_mem_len=0.,
            shift_mem_down=0,
            emb_dropout=0.,
            num_memory_tokens=None,
            tie_embedding=False,
            use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim
        emb_dim = default(emb_dim, dim)

        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down

        self.token_emb = nn.Embedding(num_tokens, emb_dim)
        self.pos_emb = AbsolutePositionalEmbedding(emb_dim, max_seq_len) if (
                    use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = nn.Linear(emb_dim, dim) if emb_dim != dim else nn.Identity()
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = nn.Linear(dim, num_tokens) if not tie_embedding else lambda t: t @ self.token_emb.weight.t()

        # memory tokens (like [cls]) from Memory Transformers paper
        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(torch.randn(num_memory_tokens, dim))

    def init_(self):
        nn.init.kaiming_normal_(self.token_emb.weight)

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_hiddens=False,
            return_attn=False,
            mems=None,
            use_cache=False,
            **kwargs
    ):
        b, n, device, num_mem = *x.shape, x.device, self.num_memory_tokens
        x = self.token_emb(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)

        x = self.project_emb(x)

        if num_mem > 0:
            mem = repeat(self.memory_tokens, 'n d -> b n d', b=b)
            x = torch.cat((mem, x), dim=1)

            # auto-handle masking after appending memory tokens
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)

        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = mems[:self.shift_mem_down], mems[self.shift_mem_down:]
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = self.to_logits(x) if not return_embeddings else x

        if return_hiddens:
            hiddens = intermediates.hiddens
            return out, hiddens

        res = [out]
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            res.append(attn_maps)
        if use_cache:
            res.append(intermediates.past_key_values)

        if len(res) > 1:
            return tuple(res)
        return res[0]


class ContinuousTransformerWrapper(nn.Module):
    def __init__(
            self,
            *,
            max_seq_len,
            attn_layers,
            dim_in=None,
            dim_out=None,
            emb_dim=None,
            emb_dropout=0.,
            use_pos_emb=True
    ):
        super().__init__()
        assert isinstance(attn_layers, AttentionLayers), 'attention layers must be one of Encoder or Decoder'

        dim = attn_layers.dim

        self.max_seq_len = max_seq_len

        self.pos_emb = AbsolutePositionalEmbedding(dim, max_seq_len) if (
                    use_pos_emb and not attn_layers.has_pos_emb) else always(0)
        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_in = nn.Linear(dim_in, dim) if exists(dim_in) else nn.Identity()

        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.project_out = nn.Linear(dim, dim_out) if exists(dim_out) else nn.Identity()

    def forward(
            self,
            x,
            return_embeddings=False,
            mask=None,
            return_attn=False,
            mems=None,
            use_cache=False,
            **kwargs
    ):
        b, n, _, device = *x.shape, x.device

        x = self.project_in(x)
        x = x + self.pos_emb(x)
        x = self.emb_dropout(x)

        x, intermediates = self.attn_layers(x, mask=mask, mems=mems, return_hiddens=True, **kwargs)
        x = self.norm(x)

        out = self.project_out(x) if not return_embeddings else x

        res = [out]
        if return_attn:
            attn_maps = list(map(lambda t: t.post_softmax_attn, intermediates.attn_intermediates))
            res.append(attn_maps)
        if use_cache:
            res.append(intermediates.past_key_values)

        if len(res) > 1:
            return tuple(res)
        return res[0]

