import torch
from coqpit import Coqpit
from torch import nn
from TTS.style_encoder.layers.gst import GST
from TTS.style_encoder.layers.vae_se import VAEStyleEncoder

class StyleEncoder(nn.Module):
    def __init__(self, config:Coqpit) -> None:
        super().__init__()
        # Load from Config
        for key in config:
            setattr(self, key, config[key])

        if self.se_type == 'gst':
            self.layer = GST(
                num_mel = self.num_mel,
                gst_embedding_dim = self.style_embedding_dim,
                num_heads = self.gst_num_heads,
                num_style_tokens = self.gst_num_style_tokens,
            )
        elif self.se_type == 'vae_se':
            self.layer = VAEStyleEncoder(
                num_mel = self.num_mel,
                embedding_dim = self.embedding_dim,
                latent_dim = self.vae_latent_dim
            )
        else:
            raise NotImplementedError

    def forward(self, inputs):
        return self.layer(inputs)