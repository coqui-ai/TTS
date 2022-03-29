import torch
from coqpit import Coqpit
from torch import nn
from TTS.style_encoder.layers.gst import GST
from TTS.style_encoder.layers.vae import VAEStyleEncoder
from TTS.style_encoder.layers.diffusion import DiffStyleEncoder

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
        elif self.se_type == 'vae':
            self.layer = VAEStyleEncoder(
                num_mel = self.num_mel,
                embedding_dim = self.embedding_dim,
                latent_dim = self.vae_latent_dim
            )
        elif self.se_type == 'diffusion':
            self.layer = DiffStyleEncoder(
                diff_num_timesteps = self.diff_num_timesteps, 
                diff_schedule_type = self.diff_schedule_type,
                diff_loss_type = self.diff_loss_type, 
                diff_use_diff_output = self.diff_use_diff_output,
                ref_online = self.diff_ref_online, 
                ref_num_mel = self.num_mel, 
                ref_style_emb_dim = self.style_embedding_dim, 
                den_step_dim = self.diff_step_dim,
                den_in_out_ch = self.diff_in_out_ch, 
                den_num_heads = self.diff_num_heads, 
                den_hidden_channels = self.diff_hidden_channels, 
                den_num_blocks = self.diff_num_blocks,
                den_dropout = self.diff_dropout
            )
        else:
            raise NotImplementedError

    def forward(self, inputs):
        if self.se_type == 'gst':
            out = self.gst_embedding(*inputs)
        elif self.se_type == 'diffusion':
            out = self.diff_forward(*inputs)
        elif self.se_type == 'vae':
            out = self.vae_forward(*inputs)
        else:
            raise NotImplementedError
        return out

    def inference(self, inputs, **kwargs):
        if self.se_type == 'gst':
            out = self.gst_embedding(inputs, kwargs['style_mel'], kwargs['d_vectors'])
        elif self.se_type == 'diffusion':
            out = self.diff_inference(inputs, mel_in = kwargs['style_mel'], infer_from = kwargs['diff_t'])
        elif self.se_type == 'vae':
            out = self.vae_inference(inputs, ref_mels= kwargs['style_mel'], z = kwargs['z'])
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

    def diff_forward(self, inputs, ref_mels):
            diff_output = self.layer.forward(ref_mels)
            return self._concat_embedding(inputs, diff_output['style']), diff_output['noises']

    def diff_inference(self, inputs, ref_mels, infer_from):
            diff_output = self.layer.inference(ref_mels, infer_from)
            return self._concat_embedding(inputs, diff_output['style'])

    def _concat_embedding(self, outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embedded_speakers_], dim=-1)
        return outputs

    def vae_forward(self, inputs, ref_mels): 
        vae_output = self.layer.foward(ref_mels)
        return self._concat_embedding(inputs, vae_output['z'])
    
    def vae_inference(self, inputs, ref_mels, z=None):
        if(z):
            return self._concat_embedding(inputs, z)  # If an specific z is passed it uses it
        else:
            vae_output = self.layer.foward(ref_mels)
            return self._concat_embedding(inputs, vae_output['z'])
            