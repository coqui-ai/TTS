import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from TTS.tts.layers.tortoise.arch_utils import AttentionBlock
from TTS.tts.layers.tortoise.xtransformers import ContinuousTransformerWrapper, Encoder


def exists(val):
    return val is not None


def masked_mean(t, mask):
    t = t.masked_fill(~mask, 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)


class CollapsingTransformer(nn.Module):
    def __init__(self, model_dim, output_dims, heads, dropout, depth, mask_percentage=0, **encoder_kwargs):
        super().__init__()
        self.transformer = ContinuousTransformerWrapper(
            max_seq_len=-1,
            use_pos_emb=False,
            attn_layers=Encoder(
                dim=model_dim,
                depth=depth,
                heads=heads,
                ff_dropout=dropout,
                ff_mult=1,
                attn_dropout=dropout,
                use_rmsnorm=True,
                ff_glu=True,
                rotary_pos_emb=True,
                **encoder_kwargs,
            ),
        )
        self.pre_combiner = nn.Sequential(
            nn.Conv1d(model_dim, output_dims, 1),
            AttentionBlock(output_dims, num_heads=heads, do_checkpoint=False),
            nn.Conv1d(output_dims, output_dims, 1),
        )
        self.mask_percentage = mask_percentage

    def forward(self, x, **transformer_kwargs):
        h = self.transformer(x, **transformer_kwargs)
        h = h.permute(0, 2, 1)
        h = self.pre_combiner(h).permute(0, 2, 1)
        if self.training:
            mask = torch.rand_like(h.float()) > self.mask_percentage
        else:
            mask = torch.ones_like(h.float()).bool()
        return masked_mean(h, mask)


class ConvFormatEmbedding(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.emb = nn.Embedding(*args, **kwargs)

    def forward(self, x):
        y = self.emb(x)
        return y.permute(0, 2, 1)


class CVVP(nn.Module):
    def __init__(
        self,
        model_dim=512,
        transformer_heads=8,
        dropout=0.1,
        conditioning_enc_depth=8,
        cond_mask_percentage=0,
        mel_channels=80,
        mel_codes=None,
        speech_enc_depth=8,
        speech_mask_percentage=0,
        latent_multiplier=1,
    ):
        super().__init__()
        latent_dim = latent_multiplier * model_dim
        self.temperature = nn.Parameter(torch.tensor(1.0))

        self.cond_emb = nn.Sequential(
            nn.Conv1d(mel_channels, model_dim // 2, kernel_size=5, stride=2, padding=2),
            nn.Conv1d(model_dim // 2, model_dim, kernel_size=3, stride=2, padding=1),
        )
        self.conditioning_transformer = CollapsingTransformer(
            model_dim,
            model_dim,
            transformer_heads,
            dropout,
            conditioning_enc_depth,
            cond_mask_percentage,
        )
        self.to_conditioning_latent = nn.Linear(latent_dim, latent_dim, bias=False)

        if mel_codes is None:
            self.speech_emb = nn.Conv1d(mel_channels, model_dim, kernel_size=5, padding=2)
        else:
            self.speech_emb = ConvFormatEmbedding(mel_codes, model_dim)
        self.speech_transformer = CollapsingTransformer(
            model_dim,
            latent_dim,
            transformer_heads,
            dropout,
            speech_enc_depth,
            speech_mask_percentage,
        )
        self.to_speech_latent = nn.Linear(latent_dim, latent_dim, bias=False)

    def get_grad_norm_parameter_groups(self):
        return {
            "conditioning": list(self.conditioning_transformer.parameters()),
            "speech": list(self.speech_transformer.parameters()),
        }

    def forward(self, mel_cond, mel_input, return_loss=False):
        cond_emb = self.cond_emb(mel_cond).permute(0, 2, 1)
        enc_cond = self.conditioning_transformer(cond_emb)
        cond_latents = self.to_conditioning_latent(enc_cond)

        speech_emb = self.speech_emb(mel_input).permute(0, 2, 1)
        enc_speech = self.speech_transformer(speech_emb)
        speech_latents = self.to_speech_latent(enc_speech)

        cond_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (cond_latents, speech_latents))
        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum("n d, n d -> n", cond_latents, speech_latents) * temp
            return sim

        sim = einsum("i d, j d -> i j", cond_latents, speech_latents) * temp
        labels = torch.arange(cond_latents.shape[0], device=mel_input.device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2

        return loss


if __name__ == "__main__":
    clvp = CVVP()
    clvp(torch.randn(2, 80, 100), torch.randn(2, 80, 95), return_loss=True)
