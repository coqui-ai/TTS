import torch
import torch.nn as nn
import numpy as np
import math
from functools import partial
from inspect import isfunction
import torch.nn.functional as F

class DiffStyleEncoder(nn.Module):
    def __init__(self, 
        diff_num_timesteps, 
        diff_schedule_type,
        diff_loss_type, 
        diff_use_diff_output,
        ref_online, 
        ref_num_mel, 
        ref_style_emb_dim, 
        den_step_dim,
        den_in_out_ch, 
        den_num_heads, 
        den_hidden_channels, 
        den_num_blocks,
        den_dropout) -> None:
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Reference Encoder
        self.ref_encoder = ReferenceEncoder(ref_num_mel, ref_style_emb_dim)
        self.online = ref_online    # Flag for wheher to train or not the Ref Encoder together
        
        # Diffusion Globals
        self.denoiser = DiffNet(ref_style_emb_dim, den_step_dim, den_in_out_ch, den_num_heads, den_hidden_channels, den_num_blocks, den_dropout)
        self.num_timesteps = int(diff_num_timesteps)
        self.loss_type = diff_loss_type
        self.use_diff_out = diff_use_diff_output

        # Betas
        if diff_schedule_type == 'linear':
            betas = linear_beta_schedule(self.num_timesteps)
        elif diff_schedule_type == 'cosine':
            betas = cosine_beta_schedule(self.num_timesteps)
        else:
            raise NotImplementedError

        # Alphas
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis = 0)               # alpha_t for all t
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])    # alpha_t-1 for all t
        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

        # Calculations for conditioned posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped', to_torch(np.log(np.maximum(posterior_variance, 1e-20)))) # log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain
        self.register_buffer('posterior_mean_coef1', to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))) # multiplies x_{0}
        self.register_buffer('posterior_mean_coef2', to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod))) # multiplies x_{t}

    # Forward: q(x_{t} | x_{0})
    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    # Reverse Conditioned: q(x_{t-1} | x_{t}, x_{0})
    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    # Reverse Learned: p(x_{t-1} | x_{t})
    def p_mean_variance(self, x, t, clip_denoised: bool):
        noise_pred = self.denoiser(x, t)
        x_recon = self.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    # Reparametrize p_mean_variance (Given x{t} -> predict x_{t-1})
    def p_sample(self, x, t, clip_denoised=False, repeat_noise=False):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(x=x, t=t, clip_denoised=clip_denoised)
        noise = noise_like(x.shape, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    # Reparametrize q(x_{t} | x_{0}) (Given x{0} -> diffuse x_{t})
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # Compute Diff Noises (Prediction and Target)
    def p_pred(self, x_start, t, noise=None, nonpadding=None):
        noise_target = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.denoiser(x_noisy, t)
        return {'noise_pred':noise_pred, 'noise_target':noise_target}

    def forward(self, mel_in):
        ret = {}
        b, *_, device = *mel_in.shape, mel_in.device

        # Ref-Mel-Spec -> Style Embedding
        if self.online:
            z = self.ref_encoder(mel_in).unsqueeze(1)
        else:
            with torch.no_grad():
                z = self.ref_encoder(mel_in).unsqueeze(1)
        ret['style'] = z
        
        # Sample a t from U[0,T] (common for the whole batch)
        t = torch.randint(0, self.num_timesteps, (1,), device=device).item() 
        t_vec = torch.full((b,), t, device=device, dtype=torch.long)

        # Initialize Noises Prediction Matrix
        noise_target = torch.randn_like(z)
        
        # Calculate Diffusion Loss in Step t -> t-1
        z_iso = z.detach().requires_grad_()
        ret['noises'] = self.p_pred(z_iso, t_vec, noise=noise_target)

        # Calculate z_{0} and pass to the TTS model for backpropagation
        if self.use_diff_out:
            # Propagate Noise in Style (z -> z_{t})
            z = self.q_sample(z, t=torch.tensor([t-1], device=device), noise = noise_target)

            # Reconstruct Style {z{t} -> z{t-1} -> ... -> z{0}}
            for i in reversed(range(0,t)):
                z = self.p_sample(z, t = torch.full((b,), i, device=device, dtype=torch.long))
            ret['style'] = z

        return ret

    @torch.no_grad()
    def inference(self, mel_in, infer_from):
        """
        # Inference
        #   1) Diffuse Ref.Style for :infer_from: steps and reconstruct
        #   2) Generate new style reconstructing from a Gaussian 
        """
        ret = {}
        b, *_, device = *mel_in.shape, mel_in.device

        # Ref-Mel-Spec -> Style Embedding
        z = self.ref_encoder(mel_in).unsqueeze(1)

        # Diffuse z on the noise chain -> x
        assert infer_from <= self.num_timesteps, "Input t for reconstrution greater than chain length"
        
        if isinstance(infer_from, int):
            t = infer_from
            x = self.q_sample(x_start=z, t=torch.tensor([t - 1], device=device).long())
        elif isinstance(infer_from, str) and infer_from == 'gaussian':
            t = self.num_timesteps
            x = torch.randn(z.shape, device=device)   
        else:
            raise NotImplementedError

        # Iterate the Denoiser for reconstruction
        for i in reversed(range(0, t)):
            x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long))
        ret['style'] = x
        return ret


###########################
#### Reference Encoder ####
###########################

class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.
    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim, batch_first=True
        )

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        return out.squeeze(0)

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height


##################
#### Denoiser ####
##################

class DiffNet(nn.Module):
    def __init__(self, style_emb_dim, step_emb_dim, in_out_channels, num_heads, hidden_channels_ffn, num_layers, dropout_p) -> None:
        super().__init__()
        dim = step_emb_dim
        self.diffusion_embedding = SinusoidalPosEmb(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim))
        self.project_input = nn.Linear(style_emb_dim + step_emb_dim, style_emb_dim)
        self.denoiser = FFTransformerBlock(in_out_channels, num_heads, hidden_channels_ffn, num_layers, dropout_p) 

    def forward(self, z, diffusion_step, mask=None, g=None):
        """
        :param z: [B, 1, STYLE_DIM]
        :param diffusion_step: [B]
        :return: [B, 1, STYLE_DIM]
        """
        t = self.diffusion_embedding(diffusion_step.unsqueeze(1))
        t = self.mlp(t)                     # [B, 1, STYLE_DIM]
        z_t = torch.cat([z, t], dim = -1)   # [B, 1, 2*STYLE_DIM] 
        z_in = self.project_input(z_t)      # [B, 1, STYLE_DIM]
        z = self.denoiser(z_in, mask, g)    
        return z

# From Coqui
class FFTransformerBlock(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn, num_layers, dropout_p):
        super().__init__()
        self.fft_layers = nn.ModuleList(
            [
                FFTransformer(
                    in_out_channels=in_out_channels,
                    num_heads=num_heads,
                    hidden_channels_ffn=hidden_channels_ffn,
                    dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask=None, g=None):  # pylint: disable=unused-argument
        """
        TODO: handle multi-speaker
        Shapes:
            - x: :math:`[B, C, T]`
            - mask:  :math:`[B, 1, T] or [B, T]`
        """
        if mask is not None and mask.ndim == 3:
            mask = mask.squeeze(1)
            # mask is negated, torch uses 1s and 0s reversely.
            mask = ~mask.bool()
        alignments = []
        for layer in self.fft_layers:
            x, align = layer(x, src_key_padding_mask=mask)
            alignments.append(align.unsqueeze(1))
        alignments = torch.cat(alignments, 1)
        return x

# From Coqui
class FFTransformer(nn.Module):
    def __init__(self, in_out_channels, num_heads, hidden_channels_ffn=1024, kernel_size_fft=3, dropout_p=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(in_out_channels, num_heads, dropout=dropout_p)

        padding = (kernel_size_fft - 1) // 2
        self.conv1 = nn.Conv1d(in_out_channels, hidden_channels_ffn, kernel_size=kernel_size_fft, padding=padding)
        self.conv2 = nn.Conv1d(hidden_channels_ffn, in_out_channels, kernel_size=kernel_size_fft, padding=padding)

        self.norm1 = nn.LayerNorm(in_out_channels)
        self.norm2 = nn.LayerNorm(in_out_channels)

        self.dropout1 = nn.Dropout(dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = src.permute(2, 0, 1)
        src2, enc_align = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src + src2)
        # T x B x D -> B x D x T
        src = src.permute(1, 2, 0)
        src2 = self.conv2(F.relu(self.conv1(src)))
        src2 = self.dropout2(src2)
        src = src + src2
        src = src.transpose(1, 2)
        src = self.norm2(src)
        src = src.transpose(1, 2)
        return src, enc_align

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

###########################
#### Variance Schedule ####
###########################

def linear_beta_schedule(timesteps, min_beta=1e-4, max_beta=0.01):
    """
    Defines the diffusion variance schedule to be linear.

    :param timesteps: The number of timesteps of the diffusion chain.
    :param max_beta: Final variance value.
    :param min_beta: Initial variance value.

    returns:
    An array containing the variance schedule.
    """
    betas = np.linspace(min_beta, max_beta, timesteps)
    return betas

def cosine_beta_schedule(timesteps, s=0.008):
    """
    Defines the diffusion variance schedule as proposed in https://arxiv.org/abs/2102.09672.

    :param timesteps: The number of timesteps of the diffusion chain.
    :param s: Small offset to prevent B_t from being too smal. sqrt(Bt) < 1/127.5 (pixel bin size).

    returns:
    An array containing the variance schedule.
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


###############
#### UTILS ####
###############

def exists(x):
    return x is not None

def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))

def noise_like(shape, device, repeat=False):
    repeat_noise = lambda: torch.randn((1, *shape[1:]), device=device).repeat(shape[0], *((1,) * (len(shape) - 1)))
    noise = lambda: torch.randn(shape, device=device)
    return repeat_noise() if repeat else noise()

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

if __name__ == '__main__':
    
    from torchviz import make_dot
    ref_in = torch.rand((32, 200, 80))
    a = DiffStyleEncoder(100,'cosine', 'l1', True, True, 80, 64, 128, 1, 1, 128, 2, 0.1)

    model = a(ref_in)['style']
    make_dot(model, params=dict(list(a.named_parameters()))).render("rnn_torchviz", format="png")