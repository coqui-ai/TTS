import torch
from coqpit import Coqpit
from torch import nn
from TTS.style_encoder.layers.gst import GST
from TTS.style_encoder.layers.ref import ReferenceEncoder
from TTS.style_encoder.layers.vae import VAEStyleEncoder
from TTS.style_encoder.layers.vaeflow import VAEFlowStyleEncoder
from TTS.style_encoder.layers.diffusion import DiffStyleEncoder

class StyleEncoder(nn.Module):
    def __init__(self, config:Coqpit) -> None:
        super().__init__()
        
        # Load from Config
        for key in config:
            setattr(self, key, config[key])
            
        # print(self.agg_norm)

        if(self.use_nonlinear_proj):
            self.nl_proj = nn.Linear(self.style_embedding_dim, self.style_embedding_dim)
            nn.init.xavier_normal_(self.nl_proj.weight) # Good init for projection

        self.dropout = nn.Dropout(p=0.5)

        if(self.use_proj_linear):
            self.proj = nn.Linear(self.style_embedding_dim, self.proj_dim)
            nn.init.xavier_normal_(self.proj.weight) # Good init for projection

        if self.se_type == 'gst':
            self.layer = GST(
                num_mel = self.num_mel,
                gst_embedding_dim = self.style_embedding_dim,
                num_heads = self.gst_num_heads,
                num_style_tokens = self.gst_num_style_tokens,
            )
        elif self.se_type == 're':
            self.layer = ReferenceEncoder(
                num_mel = self.num_mel,
                embedding_dim = self.style_embedding_dim
            )
        elif self.se_type == 'vae':
            self.layer = VAEStyleEncoder(
                num_mel = self.num_mel,
                embedding_dim = self.style_embedding_dim,
                latent_dim = self.vae_latent_dim
            )
        elif self.se_type == 'vaeflow':
            self.layer = VAEFlowStyleEncoder(
                num_mel = self.num_mel,
                style_emb_dim = self.style_embedding_dim,
                latent_dim = self.vae_latent_dim,
                intern_dim = self.vaeflow_intern_dim,
                number_of_flows = self.vaeflow_number_of_flows
            )
        elif self.se_type == 'diffusion':
            self.layer = DiffStyleEncoder(
                diff_num_timesteps = self.diff_num_timesteps, 
                diff_schedule_type = self.diff_schedule_type,
                diff_loss_type = self.diff_loss_type, 
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

    def forward(self, inputs, aux_input):
        if self.se_type == 'gst':
            out = self.gst_embedding(*inputs)
        elif self.se_type == 're':
            out = self.re_embedding(*inputs)
        elif self.se_type == 'diffusion':
            out = self.diff_forward(*inputs)
        elif self.se_type == 'vae':
            out = self.vae_forward(*inputs)
        elif self.se_type == 'vaeflow':
            out = self.vaeflow_forward(*inputs)
        else:
            raise NotImplementedError
        return out

    def inference(self, inputs, **kwargs):
        if self.se_type == 'gst':
            out = self.gst_embedding(inputs, kwargs['style_mel'], kwargs['d_vectors'])
        elif self.se_type == 're':
            out = self.re_embedding(inputs, kwargs['style_mel'])
        elif self.se_type == 'diffusion':
            out = self.diff_inference(inputs, ref_mels = kwargs['style_mel'], infer_from = kwargs['diff_t'])
        elif self.se_type == 'vae':
            out = self.vae_inference(inputs, ref_mels= kwargs['style_mel'], z = kwargs['z'])
        elif self.se_type == 'vaeflow':
            out = self.vaeflow_inference(inputs, ref_mels = kwargs['style_mel'], z = kwargs['z'])
        else:
            raise NotImplementedError
        return out

    def get_embedding(self, ref_mels):
        return self.layer(ref_mels)

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
            
            if(self.use_nonlinear_proj):
                # gst_outputs = self.nl_proj(gst_outputs)
                gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                gst_outputs = self.dropout(gst_outputs)
               
            if(self.use_proj_linear):
                gst_outputs = self.proj(gst_outputs)

            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(inputs, gst_outputs)
            else:
                inputs = self._add_speaker_embedding(inputs, gst_outputs)
            return inputs



    def re_embedding(self, inputs, style_input):
            if style_input is None:
                # ignore style token and return zero tensor
                gst_outputs = torch.zeros(1, 1, self.style_embedding_dim).type_as(inputs)
            elif style_input.shape[-1] == self.style_embedding_dim:
                gst_outputs = style_input
            else:
                # compute style tokens
                input_args = [style_input]
                gst_outputs = self.layer(*input_args)  # pylint: disable=not-callable
            
                if(self.use_nonlinear_proj):
                    gst_outputs = torch.tanh(self.nl_proj(gst_outputs))
                    gst_outputs = self.dropout(gst_outputs)
                
                if(self.use_proj_linear):
                    gst_outputs = self.proj(gst_outputs)

            if(self.agg_type == 'concat'):
                inputs = self._concat_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            elif(self.agg_type == 'adain'):
                inputs = self._adain(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            else:
                inputs = self._add_speaker_embedding(outputs = inputs, embedded_speakers = gst_outputs.unsqueeze(1))
            return inputs, gst_outputs

    def diff_forward(self, inputs, ref_mels):
            diff_output = self.layer.forward(ref_mels)

            if(self.use_nonlinear_proj):
                # diff_output = self.nl_proj(diff_output)
                diff_output = torch.tanh(self.nl_proj(diff_output))
                diff_output = self.dropout(diff_output)

            if(self.use_proj_linear):
                diff_output = self.proj(diff_output)

            if(self.agg_type == 'concat'):
                return self._concat_embedding(inputs, diff_output['style']), diff_output['noises']
            else:
                return self._add_speaker_embedding(inputs, diff_output['style']), diff_output['noises']

    def diff_inference(self, inputs, ref_mels, infer_from):
            diff_output = self.layer.inference(ref_mels, infer_from)
            
            if(self.use_nonlinear_proj):
                diff_output = torch.tanh(self.nl_proj(diff_output))
                diff_output = self.dropout(diff_output)
               
            if(self.use_proj_linear):
                diff_output = self.proj(diff_output)

            if(self.agg_type == 'concat'):
                return self._concat_embedding(outputs = inputs, embedded_speakers = diff_output['style'])
            else:
                return self._add_speaker_embedding(outputs = inputs, embedded_speakers = diff_output['style'])
    

    def vae_forward(self, inputs, ref_mels): 
        vae_output = self.layer.forward(ref_mels)

        if(self.use_nonlinear_proj):
            # vae_output = self.nl_proj(vae_output)
            vae_output = torch.tanh(self.nl_proj(vae_output))
            vae_output = self.dropout(vae_output)

        if(self.use_proj_linear):
            vae_output = self.proj(vae_output)

        if(self.agg_type == 'concat'):
            return self._concat_embedding(inputs, vae_output['z']), {'mean': vae_output['mean'], 'log_var' : vae_output['log_var']}
        else:
            return self._add_speaker_embedding(inputs, vae_output['z']), {'mean': vae_output['mean'], 'log_var' : vae_output['log_var']}
    
    def vae_inference(self, inputs, ref_mels, z=None):
        if(z):
            if(self.agg_type == 'concat'):
                return self._concat_embedding(inputs, z)  # If an specific z is passed it uses it
            else:
                return self._add_speaker_embedding(inputs, z)
        else:
            vae_output = self.layer.forward(ref_mels)

            if(self.use_nonlinear_proj):
                vae_output = torch.tanh(self.nl_proj(vae_output))
                vae_output = self.dropout(vae_output)
                
            if(self.use_proj_linear):
                vae_output = self.proj(vae_output)

            if(self.agg_type == 'concat'):
                return self._concat_embedding(inputs, vae_output['z'])
            else:
                return self._add_speaker_embedding(inputs, vae_output['z'])

    def vaeflow_forward(self, inputs, ref_mels): 
        vaeflow_output = self.layer.forward(ref_mels)

        if(self.use_nonlinear_proj):
            # vaeflow_output = self.nl_proj(vaeflow_output)
            vaeflow_output = torch.tanh(self.nl_proj(vaeflow_output))
            vaeflow_output = self.dropout(vaeflow_output)

        if(self.use_proj_linear):
            vaeflow_output = self.proj(vaeflow_output)

        if(self.agg_type == 'concat'):
            return self._concat_embedding(inputs, vaeflow_output['z_T']), {'mean': vaeflow_output['mean'], 'log_var' : vaeflow_output['log_var'], 'z_0' : vaeflow_output['z_0'], 'z_T' : vaeflow_output['z_T']}
        else:
            return self._add_speaker_embedding(inputs, vaeflow_output['z_T']), {'mean': vaeflow_output['mean'], 'log_var' : vaeflow_output['log_var'], 'z_0' : vaeflow_output['z_0'], 'z_T' : vaeflow_output['z_T']}

    def vaeflow_inference(self, inputs, ref_mels, z=None):
        if(z):
            if(self.agg_type == 'concat'):
                return self._concat_embedding(inputs, z)  # If an specific z is passed it uses it
            else:
                return self._add_speaker_embedding(inputs, z)
        else:
            vaeflow_output = self.layer.forward(ref_mels)

            if(self.use_nonlinear_proj):
                vaeflow_output = torch.tanh(self.nl_proj(vaeflow_output))
                vaeflow_output = self.dropout(vaeflow_output)
                
            if(self.use_proj_linear):
                vaeflow_output = self.proj(vaeflow_output)

            if(self.agg_type == 'concat'):
                return self._concat_embedding(inputs, vaeflow_output['z_T'])
            else:
                return self._add_speaker_embedding(inputs, vaeflow_output['z_T'])


    # For this two below, remember if B is batch size, L the sequence length, E is the embedding dim and D the style embed dim
    # for tacotron2 the encoder outputs comes [B,L,E] and faspitch comes [B,E,L]
    # @classmethod
    def _concat_embedding(self, outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embedded_speakers_], dim=-1)
        return outputs

    # @classmethod
    def _add_speaker_embedding(self, outputs, embedded_speakers):
        # Fixed to the forwardtts, now, for adding we normalize by l2 norm
        if(self.agg_norm == True):
            embedded_speakers_ = nn.functional.normalize(embedded_speakers).expand(outputs.size(0), outputs.size(1), -1)
        else:
            embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = outputs + embedded_speakers_
        return outputs

    # we assume that the encoder "outputs" will get the shape [B,L,E], so we mean over L and apply adain in it
    def _adain(self, outputs, embedded_speakers):

        mean_content = torch.mean(outputs, dim= [-1])
        std_content = torch.std(outputs, dim= [-1]) + 1e-5

        # print('embed_speakers shape : ' , embedded_speakers.shape)

        # embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), outputs.size(2))
        
        # print('embed_speakers_ shape : ' , embedded_speakers_.shape)

        mean_style = torch.mean(embedded_speakers, dim= [-1])
        std_style = torch.std(embedded_speakers, dim= [-1]) + 1e-5

        # print(mean_style.shape, std_style.shape, mean_style, std_style)

        # print(mean_content.shape, std_content.shape, mean_content, std_content)
        
        mean_style = mean_style.unsqueeze(1).expand(outputs.size(0), outputs.size(1), outputs.size(2))
        std_style = std_style.unsqueeze(1).expand(outputs.size(0), outputs.size(1), outputs.size(2))

        mean_content = mean_content.unsqueeze(2).expand(outputs.size(0), outputs.size(1), outputs.size(2))
        std_content = std_content.unsqueeze(2).expand(outputs.size(0), outputs.size(1), outputs.size(2))

        # print(mean_style.shape, std_style.shape, mean_style, std_style)

        # print(mean_content.shape, std_content.shape, mean_content, std_content)

        # add verbose to debug nan errors: the hypothesis is that the mean becomes high
        # print(mean_content, std_content, mean_style, std_style) # Apparently the error was dividing by 0, added 1e-5

        return (outputs - mean_content)/std_content*std_style + mean_style