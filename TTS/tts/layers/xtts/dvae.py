import functools
from math import sqrt

import torch
import torch.distributed as distributed
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from einops import rearrange


def default(val, d):
    return val if val is not None else d


def eval_decorator(fn):
    def inner(model, *args, **kwargs):
        was_training = model.training
        model.eval()
        out = fn(model, *args, **kwargs)
        model.train(was_training)
        return out

    return inner


def dvae_wav_to_mel(
    wav, mel_norms_file="../experiments/clips_mel_norms.pth", mel_norms=None, device=torch.device("cpu")
):
    mel_stft = torchaudio.transforms.MelSpectrogram(
        n_fft=1024,
        hop_length=256,
        win_length=1024,
        power=2,
        normalized=False,
        sample_rate=22050,
        f_min=0,
        f_max=8000,
        n_mels=80,
        norm="slaney",
    ).to(device)
    wav = wav.to(device)
    mel = mel_stft(wav)
    mel = torch.log(torch.clamp(mel, min=1e-5))
    if mel_norms is None:
        mel_norms = torch.load(mel_norms_file, map_location=device)
    mel = mel / mel_norms.unsqueeze(0).unsqueeze(-1)
    return mel


class Quantize(nn.Module):
    def __init__(self, dim, n_embed, decay=0.99, eps=1e-5, balancing_heuristic=False, new_return_order=False):
        super().__init__()

        self.dim = dim
        self.n_embed = n_embed
        self.decay = decay
        self.eps = eps

        self.balancing_heuristic = balancing_heuristic
        self.codes = None
        self.max_codes = 64000
        self.codes_full = False
        self.new_return_order = new_return_order

        embed = torch.randn(dim, n_embed)
        self.register_buffer("embed", embed)
        self.register_buffer("cluster_size", torch.zeros(n_embed))
        self.register_buffer("embed_avg", embed.clone())

    def forward(self, input, return_soft_codes=False):
        if self.balancing_heuristic and self.codes_full:
            h = torch.histc(self.codes, bins=self.n_embed, min=0, max=self.n_embed) / len(self.codes)
            mask = torch.logical_or(h > 0.9, h < 0.01).unsqueeze(1)
            ep = self.embed.permute(1, 0)
            ea = self.embed_avg.permute(1, 0)
            rand_embed = torch.randn_like(ep) * mask
            self.embed = (ep * ~mask + rand_embed).permute(1, 0)
            self.embed_avg = (ea * ~mask + rand_embed).permute(1, 0)
            self.cluster_size = self.cluster_size * ~mask.squeeze()
            if torch.any(mask):
                print(f"Reset {torch.sum(mask)} embedding codes.")
                self.codes = None
                self.codes_full = False

        flatten = input.reshape(-1, self.dim)
        dist = flatten.pow(2).sum(1, keepdim=True) - 2 * flatten @ self.embed + self.embed.pow(2).sum(0, keepdim=True)
        soft_codes = -dist
        _, embed_ind = soft_codes.max(1)
        embed_onehot = F.one_hot(embed_ind, self.n_embed).type(flatten.dtype)
        embed_ind = embed_ind.view(*input.shape[:-1])
        quantize = self.embed_code(embed_ind)

        if self.balancing_heuristic:
            if self.codes is None:
                self.codes = embed_ind.flatten()
            else:
                self.codes = torch.cat([self.codes, embed_ind.flatten()])
                if len(self.codes) > self.max_codes:
                    self.codes = self.codes[-self.max_codes :]
                    self.codes_full = True

        if self.training:
            embed_onehot_sum = embed_onehot.sum(0)
            embed_sum = flatten.transpose(0, 1) @ embed_onehot

            if distributed.is_initialized() and distributed.get_world_size() > 1:
                distributed.all_reduce(embed_onehot_sum)
                distributed.all_reduce(embed_sum)

            self.cluster_size.data.mul_(self.decay).add_(embed_onehot_sum, alpha=1 - self.decay)
            self.embed_avg.data.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)
            n = self.cluster_size.sum()
            cluster_size = (self.cluster_size + self.eps) / (n + self.n_embed * self.eps) * n
            embed_normalized = self.embed_avg / cluster_size.unsqueeze(0)
            self.embed.data.copy_(embed_normalized)

        diff = (quantize.detach() - input).pow(2).mean()
        quantize = input + (quantize - input).detach()

        if return_soft_codes:
            return quantize, diff, embed_ind, soft_codes.view(input.shape[:-1] + (-1,))
        elif self.new_return_order:
            return quantize, embed_ind, diff
        else:
            return quantize, diff, embed_ind

    def embed_code(self, embed_id):
        return F.embedding(embed_id, self.embed.transpose(0, 1))


# Fits a soft-discretized input to a normal-PDF across the specified dimension.
# In other words, attempts to force the discretization function to have a mean equal utilization across all discrete
# values with the specified expected variance.
class DiscretizationLoss(nn.Module):
    def __init__(self, discrete_bins, dim, expected_variance, store_past=0):
        super().__init__()
        self.discrete_bins = discrete_bins
        self.dim = dim
        self.dist = torch.distributions.Normal(0, scale=expected_variance)
        if store_past > 0:
            self.record_past = True
            self.register_buffer("accumulator_index", torch.zeros(1, dtype=torch.long, device="cpu"))
            self.register_buffer("accumulator_filled", torch.zeros(1, dtype=torch.long, device="cpu"))
            self.register_buffer("accumulator", torch.zeros(store_past, discrete_bins))
        else:
            self.record_past = False

    def forward(self, x):
        other_dims = set(range(len(x.shape))) - set([self.dim])
        averaged = x.sum(dim=tuple(other_dims)) / x.sum()
        averaged = averaged - averaged.mean()

        if self.record_past:
            acc_count = self.accumulator.shape[0]
            avg = averaged.detach().clone()
            if self.accumulator_filled > 0:
                averaged = torch.mean(self.accumulator, dim=0) * (acc_count - 1) / acc_count + averaged / acc_count

            # Also push averaged into the accumulator.
            self.accumulator[self.accumulator_index] = avg
            self.accumulator_index += 1
            if self.accumulator_index >= acc_count:
                self.accumulator_index *= 0
                if self.accumulator_filled <= 0:
                    self.accumulator_filled += 1

        return torch.sum(-self.dist.log_prob(averaged))


class ResBlock(nn.Module):
    def __init__(self, chan, conv, activation):
        super().__init__()
        self.net = nn.Sequential(
            conv(chan, chan, 3, padding=1),
            activation(),
            conv(chan, chan, 3, padding=1),
            activation(),
            conv(chan, chan, 1),
        )

    def forward(self, x):
        return self.net(x) + x


class UpsampledConv(nn.Module):
    def __init__(self, conv, *args, **kwargs):
        super().__init__()
        assert "stride" in kwargs.keys()
        self.stride = kwargs["stride"]
        del kwargs["stride"]
        self.conv = conv(*args, **kwargs)

    def forward(self, x):
        up = nn.functional.interpolate(x, scale_factor=self.stride, mode="nearest")
        return self.conv(up)


# DiscreteVAE partially derived from lucidrains DALLE implementation
# Credit: https://github.com/lucidrains/DALLE-pytorch
class DiscreteVAE(nn.Module):
    def __init__(
        self,
        positional_dims=2,
        num_tokens=512,
        codebook_dim=512,
        num_layers=3,
        num_resnet_blocks=0,
        hidden_dim=64,
        channels=3,
        stride=2,
        kernel_size=4,
        use_transposed_convs=True,
        encoder_norm=False,
        activation="relu",
        smooth_l1_loss=False,
        straight_through=False,
        normalization=None,  # ((0.5,) * 3, (0.5,) * 3),
        record_codes=False,
        discretization_loss_averaging_steps=100,
        lr_quantizer_args={},
    ):
        super().__init__()
        has_resblocks = num_resnet_blocks > 0

        self.num_tokens = num_tokens
        self.num_layers = num_layers
        self.straight_through = straight_through
        self.positional_dims = positional_dims
        self.discrete_loss = DiscretizationLoss(
            num_tokens, 2, 1 / (num_tokens * 2), discretization_loss_averaging_steps
        )

        assert positional_dims > 0 and positional_dims < 3  # This VAE only supports 1d and 2d inputs for now.
        if positional_dims == 2:
            conv = nn.Conv2d
            conv_transpose = nn.ConvTranspose2d
        else:
            conv = nn.Conv1d
            conv_transpose = nn.ConvTranspose1d
        if not use_transposed_convs:
            conv_transpose = functools.partial(UpsampledConv, conv)

        if activation == "relu":
            act = nn.ReLU
        elif activation == "silu":
            act = nn.SiLU
        else:
            assert NotImplementedError()

        enc_layers = []
        dec_layers = []

        if num_layers > 0:
            enc_chans = [hidden_dim * 2**i for i in range(num_layers)]
            dec_chans = list(reversed(enc_chans))

            enc_chans = [channels, *enc_chans]

            dec_init_chan = codebook_dim if not has_resblocks else dec_chans[0]
            dec_chans = [dec_init_chan, *dec_chans]

            enc_chans_io, dec_chans_io = map(lambda t: list(zip(t[:-1], t[1:])), (enc_chans, dec_chans))

            pad = (kernel_size - 1) // 2
            for (enc_in, enc_out), (dec_in, dec_out) in zip(enc_chans_io, dec_chans_io):
                enc_layers.append(nn.Sequential(conv(enc_in, enc_out, kernel_size, stride=stride, padding=pad), act()))
                if encoder_norm:
                    enc_layers.append(nn.GroupNorm(8, enc_out))
                dec_layers.append(
                    nn.Sequential(conv_transpose(dec_in, dec_out, kernel_size, stride=stride, padding=pad), act())
                )
            dec_out_chans = dec_chans[-1]
            innermost_dim = dec_chans[0]
        else:
            enc_layers.append(nn.Sequential(conv(channels, hidden_dim, 1), act()))
            dec_out_chans = hidden_dim
            innermost_dim = hidden_dim

        for _ in range(num_resnet_blocks):
            dec_layers.insert(0, ResBlock(innermost_dim, conv, act))
            enc_layers.append(ResBlock(innermost_dim, conv, act))

        if num_resnet_blocks > 0:
            dec_layers.insert(0, conv(codebook_dim, innermost_dim, 1))

        enc_layers.append(conv(innermost_dim, codebook_dim, 1))
        dec_layers.append(conv(dec_out_chans, channels, 1))

        self.encoder = nn.Sequential(*enc_layers)
        self.decoder = nn.Sequential(*dec_layers)

        self.loss_fn = F.smooth_l1_loss if smooth_l1_loss else F.mse_loss
        self.codebook = Quantize(codebook_dim, num_tokens, new_return_order=True)

        # take care of normalization within class
        self.normalization = normalization
        self.record_codes = record_codes
        if record_codes:
            self.codes = torch.zeros((1228800,), dtype=torch.long)
            self.code_ind = 0
            self.total_codes = 0
        self.internal_step = 0

    def norm(self, images):
        if not self.normalization is not None:
            return images

        means, stds = map(lambda t: torch.as_tensor(t).to(images), self.normalization)
        arrange = "c -> () c () ()" if self.positional_dims == 2 else "c -> () c ()"
        means, stds = map(lambda t: rearrange(t, arrange), (means, stds))
        images = images.clone()
        images.sub_(means).div_(stds)
        return images

    def get_debug_values(self, step, __):
        if self.record_codes and self.total_codes > 0:
            # Report annealing schedule
            return {"histogram_codes": self.codes[: self.total_codes]}
        else:
            return {}

    @torch.no_grad()
    @eval_decorator
    def get_codebook_indices(self, images):
        img = self.norm(images)
        logits = self.encoder(img).permute((0, 2, 3, 1) if len(img.shape) == 4 else (0, 2, 1))
        sampled, codes, _ = self.codebook(logits)
        self.log_codes(codes)
        return codes

    def decode(self, img_seq):
        self.log_codes(img_seq)
        if hasattr(self.codebook, "embed_code"):
            image_embeds = self.codebook.embed_code(img_seq)
        else:
            image_embeds = F.embedding(img_seq, self.codebook.codebook)
        b, n, d = image_embeds.shape

        kwargs = {}
        if self.positional_dims == 1:
            arrange = "b n d -> b d n"
        else:
            h = w = int(sqrt(n))
            arrange = "b (h w) d -> b d h w"
            kwargs = {"h": h, "w": w}
        image_embeds = rearrange(image_embeds, arrange, **kwargs)
        images = [image_embeds]
        for layer in self.decoder:
            images.append(layer(images[-1]))
        return images[-1], images[-2]

    def infer(self, img):
        img = self.norm(img)
        logits = self.encoder(img).permute((0, 2, 3, 1) if len(img.shape) == 4 else (0, 2, 1))
        sampled, codes, commitment_loss = self.codebook(logits)
        return self.decode(codes)

    # Note: This module is not meant to be run in forward() except while training. It has special logic which performs
    # evaluation using quantized values when it detects that it is being run in eval() mode, which will be substantially
    # more lossy (but useful for determining network performance).
    def forward(self, img):
        img = self.norm(img)
        logits = self.encoder(img).permute((0, 2, 3, 1) if len(img.shape) == 4 else (0, 2, 1))
        sampled, codes, commitment_loss = self.codebook(logits)
        sampled = sampled.permute((0, 3, 1, 2) if len(img.shape) == 4 else (0, 2, 1))

        if self.training:
            out = sampled
            for d in self.decoder:
                out = d(out)
            self.log_codes(codes)
        else:
            # This is non-differentiable, but gives a better idea of how the network is actually performing.
            out, _ = self.decode(codes)

        # reconstruction loss
        recon_loss = self.loss_fn(img, out, reduction="none")

        return recon_loss, commitment_loss, out

    def log_codes(self, codes):
        # This is so we can debug the distribution of codes being learned.
        if self.record_codes and self.internal_step % 10 == 0:
            codes = codes.flatten()
            l = codes.shape[0]
            i = self.code_ind if (self.codes.shape[0] - self.code_ind) > l else self.codes.shape[0] - l
            self.codes[i : i + l] = codes.cpu()
            self.code_ind = self.code_ind + l
            if self.code_ind >= self.codes.shape[0]:
                self.code_ind = 0
            self.total_codes += 1
        self.internal_step += 1
