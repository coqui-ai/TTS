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
        if self.se_type == 'gst':
            out = self.gst_embedding(*inputs)
        else:
            raise NotImplementedError
        return out

    def gst_embedding(self, inputs, style_input, speaker_embedding=None):
            if isinstance(style_input, dict):
                # multiply each style token with a weight
                query = torch.zeros(1, 1, self.style_embedding_dim // 2).type_as(inputs)
                if speaker_embedding is not None:
                    query = torch.cat([query, speaker_embedding.reshape(1, 1, -1)], dim=-1)

                _GST = torch.tanh(self.layer.style_token_layer.style_tokens)
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
                for k_token, v_amplifier in style_input.items():
                    key = _GST[int(k_token)].unsqueeze(0).expand(1, -1, -1)
                    gst_outputs_att = self.layer.style_token_layer.attention(query, key)
                    gst_outputs = gst_outputs + gst_outputs_att * v_amplifier
            elif style_input is None:
                # ignore style token and return zero tensor
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
            else:
                # compute style tokens
                input_args = [style_input, speaker_embedding]
                gst_outputs = self.layer(*input_args)  # pylint: disable=not-callable
            inputs = self._concat_embedding(inputs, gst_outputs)
            return inputs

    def _concat_embedding(self, outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embedded_speakers_], dim=-1)
        return outputs