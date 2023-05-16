import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn

# helpers


def exists(val):
    return val is not None


def default(val, d):
    return val if exists(val) else d


def cast_tuple(val, depth=1):
    if isinstance(val, list):
        val = tuple(val)
    return val if isinstance(val, tuple) else (val,) * depth


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def stable_softmax(t, dim=-1, alpha=32**2):
    t = t / alpha
    t = t - torch.amax(t, dim=dim, keepdim=True).detach()
    return (t * alpha).softmax(dim=dim)


def route_args(router, args, depth):
    routed_args = [(dict(), dict()) for _ in range(depth)]
    matched_keys = [key for key in args.keys() if key in router]

    for key in matched_keys:
        val = args[key]
        for depth, ((f_args, g_args), routes) in enumerate(zip(routed_args, router[key])):
            new_f_args, new_g_args = map(lambda route: ({key: val} if route else {}), routes)
            routed_args[depth] = ({**f_args, **new_f_args}, {**g_args, **new_g_args})
    return routed_args


# classes
class SequentialSequence(nn.Module):
    def __init__(self, layers, args_route={}, layer_dropout=0.0):
        super().__init__()
        assert all(
            len(route) == len(layers) for route in args_route.values()
        ), "each argument route map must have the same depth as the number of sequential layers"
        self.layers = layers
        self.args_route = args_route
        self.layer_dropout = layer_dropout

    def forward(self, x, **kwargs):
        args = route_args(self.args_route, kwargs, len(self.layers))
        layers_and_args = list(zip(self.layers, args))

        for (f, g), (f_args, g_args) in layers_and_args:
            x = x + f(x, **f_args)
            x = x + g(x, **g_args)
        return x


class DivideMax(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        maxes = x.amax(dim=self.dim, keepdim=True).detach()
        return x / maxes


# https://arxiv.org/abs/2103.17239
class LayerScale(nn.Module):
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) * self.scale


# layer norm


class PreNorm(nn.Module):
    def __init__(self, dim, fn, sandwich=False):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm_out = nn.LayerNorm(dim) if sandwich else nn.Identity()
        self.fn = fn

    def forward(self, x, **kwargs):
        x = self.norm(x)
        x = self.fn(x, **kwargs)
        return self.norm_out(x)


# feed forward


class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    def __init__(self, dim, dropout=0.0, mult=4.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
        )

    def forward(self, x):
        return self.net(x)


# Attention


class Attention(nn.Module):
    def __init__(self, dim, seq_len, causal=True, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.seq_len = seq_len
        self.scale = dim_head**-0.5

        self.causal = causal

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))

    def forward(self, x, mask=None):
        b, n, _, h, device = *x.shape, self.heads, x.device
        softmax = torch.softmax

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=h), qkv)

        q = q * self.scale

        dots = torch.einsum("b h i d, b h j d -> b h i j", q, k)
        mask_value = max_neg_value(dots)

        if exists(mask):
            mask = rearrange(mask, "b j -> b () () j")
            dots.masked_fill_(~mask, mask_value)
            del mask

        if self.causal:
            i, j = dots.shape[-2:]
            mask = torch.ones(i, j, device=device).triu_(j - i + 1).bool()
            dots.masked_fill_(mask, mask_value)

        attn = softmax(dots, dim=-1)

        out = torch.einsum("b h i j, b h j d -> b h i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        out = self.to_out(out)
        return out


# main transformer class
class Transformer(nn.Module):
    def __init__(
        self,
        *,
        dim,
        depth,
        seq_len,
        causal=True,
        heads=8,
        dim_head=64,
        ff_mult=4,
        attn_dropout=0.0,
        ff_dropout=0.0,
        sparse_attn=False,
        sandwich_norm=False,
    ):
        super().__init__()
        layers = nn.ModuleList([])
        sparse_layer = cast_tuple(sparse_attn, depth)

        for ind, sparse_attn in zip(range(depth), sparse_layer):
            attn = Attention(
                dim,
                causal=causal,
                seq_len=seq_len,
                heads=heads,
                dim_head=dim_head,
                dropout=attn_dropout,
            )

            ff = FeedForward(dim, mult=ff_mult, dropout=ff_dropout)

            layers.append(
                nn.ModuleList(
                    [
                        LayerScale(dim, ind + 1, PreNorm(dim, attn, sandwich=sandwich_norm)),
                        LayerScale(dim, ind + 1, PreNorm(dim, ff, sandwich=sandwich_norm)),
                    ]
                )
            )

        execute_type = SequentialSequence
        route_attn = ((True, False),) * depth
        attn_route_map = {"mask": route_attn}

        self.layers = execute_type(layers, args_route=attn_route_map)

    def forward(self, x, **kwargs):
        return self.layers(x, **kwargs)
