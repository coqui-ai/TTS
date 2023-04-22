import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import einsum

from TTS.tts.layers.tortoise.arch_utils import CheckpointedXTransformerEncoder
from TTS.tts.layers.tortoise.transformer import Transformer
from TTS.tts.layers.tortoise.xtransformers import Encoder


def exists(val):
    return val is not None


def masked_mean(t, mask, dim=1):
    t = t.masked_fill(~mask[:, :, None], 0.0)
    return t.sum(dim=1) / mask.sum(dim=1)[..., None]


class CLVP(nn.Module):
    """
    CLIP model retrofitted for performing contrastive evaluation between tokenized audio data and the corresponding
    transcribed text.

    Originally from https://github.com/lucidrains/DALLE-pytorch/blob/main/dalle_pytorch/dalle_pytorch.py
    """

    def __init__(
        self,
        *,
        dim_text=512,
        dim_speech=512,
        dim_latent=512,
        num_text_tokens=256,
        text_enc_depth=6,
        text_seq_len=120,
        text_heads=8,
        num_speech_tokens=8192,
        speech_enc_depth=6,
        speech_heads=8,
        speech_seq_len=250,
        text_mask_percentage=0,
        voice_mask_percentage=0,
        wav_token_compression=1024,
        use_xformers=False,
    ):
        super().__init__()
        self.text_emb = nn.Embedding(num_text_tokens, dim_text)
        self.to_text_latent = nn.Linear(dim_text, dim_latent, bias=False)

        self.speech_emb = nn.Embedding(num_speech_tokens, dim_speech)
        self.to_speech_latent = nn.Linear(dim_speech, dim_latent, bias=False)

        if use_xformers:
            self.text_transformer = CheckpointedXTransformerEncoder(
                needs_permute=False,
                exit_permute=False,
                max_seq_len=-1,
                attn_layers=Encoder(
                    dim=dim_text,
                    depth=text_enc_depth,
                    heads=text_heads,
                    ff_dropout=0.1,
                    ff_mult=2,
                    attn_dropout=0.1,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ),
            )
            self.speech_transformer = CheckpointedXTransformerEncoder(
                needs_permute=False,
                exit_permute=False,
                max_seq_len=-1,
                attn_layers=Encoder(
                    dim=dim_speech,
                    depth=speech_enc_depth,
                    heads=speech_heads,
                    ff_dropout=0.1,
                    ff_mult=2,
                    attn_dropout=0.1,
                    use_rmsnorm=True,
                    ff_glu=True,
                    rotary_pos_emb=True,
                ),
            )
        else:
            self.text_transformer = Transformer(
                causal=False, seq_len=text_seq_len, dim=dim_text, depth=text_enc_depth, heads=text_heads
            )
            self.speech_transformer = Transformer(
                causal=False, seq_len=speech_seq_len, dim=dim_speech, depth=speech_enc_depth, heads=speech_heads
            )

        self.temperature = nn.Parameter(torch.tensor(1.0))
        self.text_mask_percentage = text_mask_percentage
        self.voice_mask_percentage = voice_mask_percentage
        self.wav_token_compression = wav_token_compression
        self.xformers = use_xformers
        if not use_xformers:
            self.text_pos_emb = nn.Embedding(text_seq_len, dim_text)
            self.speech_pos_emb = nn.Embedding(num_speech_tokens, dim_speech)

    def forward(self, text, speech_tokens, return_loss=False):
        b, device = text.shape[0], text.device
        if self.training:
            text_mask = torch.rand_like(text.float()) > self.text_mask_percentage
            voice_mask = torch.rand_like(speech_tokens.float()) > self.voice_mask_percentage
        else:
            text_mask = torch.ones_like(text.float()).bool()
            voice_mask = torch.ones_like(speech_tokens.float()).bool()

        text_emb = self.text_emb(text)
        speech_emb = self.speech_emb(speech_tokens)

        if not self.xformers:
            text_emb += self.text_pos_emb(torch.arange(text.shape[1], device=device))
            speech_emb += self.speech_pos_emb(torch.arange(speech_emb.shape[1], device=device))

        enc_text = self.text_transformer(text_emb, mask=text_mask)
        enc_speech = self.speech_transformer(speech_emb, mask=voice_mask)

        text_latents = masked_mean(enc_text, text_mask, dim=1)
        speech_latents = masked_mean(enc_speech, voice_mask, dim=1)

        text_latents = self.to_text_latent(text_latents)
        speech_latents = self.to_speech_latent(speech_latents)

        text_latents, speech_latents = map(lambda t: F.normalize(t, p=2, dim=-1), (text_latents, speech_latents))

        temp = self.temperature.exp()

        if not return_loss:
            sim = einsum("n d, n d -> n", text_latents, speech_latents) * temp
            return sim

        sim = einsum("i d, j d -> i j", text_latents, speech_latents) * temp
        labels = torch.arange(b, device=device)
        loss = (F.cross_entropy(sim, labels) + F.cross_entropy(sim.t(), labels)) / 2
        return loss


if __name__ == "__main__":
    clip = CLVP(text_mask_percentage=0.2, voice_mask_percentage=0.2)
    clip(
        torch.randint(0, 256, (2, 120)),
        torch.tensor([50, 100]),
        torch.randint(0, 8192, (2, 250)),
        torch.tensor([101, 102]),
        return_loss=True,
    )
    nonloss = clip(
        torch.randint(0, 256, (2, 120)),
        torch.tensor([50, 100]),
        torch.randint(0, 8192, (2, 250)),
        torch.tensor([101, 102]),
        return_loss=False,
    )
    print(nonloss.shape)
