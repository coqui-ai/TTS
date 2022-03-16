from importlib.util import spec_from_file_location
import torch
import torch.nn as nn
import numpy as np
import math
from math import sqrt
from functools import partial
from inspect import isfunction
import torch.nn.functional as F

class DiffStyleEncoder(nn.Module):
    def __init__(self, 
        diff_num_timesteps = 1000, 
        diff_schedule_type = 'cosine', 
        diff_K_step = 51, 
        diff_loss_type = 'l1', 
        ref_online = True, 
        ref_num_mel = 80, 
        ref_style_emb_dim = 64, 
        den_in_dims=1, 
        den_num_residual_layers=20, 
        den_residual_channels=256, 
        den_dilation_cycle_length=1) -> None:
        super().__init__()
        to_torch = partial(torch.tensor, dtype=torch.float32)

        # Reference Encoder
        self.ref_encoder = ReferenceEncoder(ref_num_mel, ref_style_emb_dim)
        self.online = ref_online    # Flag for wheher to train or not the Ref Encoder together

        # Diffusion Globals
        self.denoiser = DiffNet(den_in_dims, den_num_residual_layers, den_residual_channels, den_dilation_cycle_length)
        self.num_timesteps = int(diff_num_timesteps)
        self.K_step = diff_K_step
        self.loss_type = diff_loss_type

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

    # x_{0} = f(x_{t}, ε), ε ~ N(0,I)
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
    @torch.no_grad()
    def p_sample(self, x, t, clip_denoised=True, repeat_noise=False):
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

    # Compute Loss between the pred noise and real noise
    def p_losses(self, x_start, t, noise=None, nonpadding=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        noise_pred = self.denoiser(x_noisy, t)

        if self.loss_type == 'l1':
            if nonpadding is not None:
                loss = ((noise - noise_pred).abs() * nonpadding.unsqueeze(1)).mean()
            else:
                # print('are you sure w/o nonpadding?')
                loss = (noise - noise_pred).abs().mean()
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, noise_pred)
        else:
            raise NotImplementedError()
        return loss

    def forward(self, mel_in, infer, reconstruct_from):
        ret = {}
        b, *_, device = *mel_in.shape, mel_in.device

        # Get Style Embedding
        if self.online:
            ret['latent'] = self.ref_encoder(mel_in).unsqueeze(1)
        else:
            with torch.no_grad():
                ret['latent'] = self.ref_encoder(mel_in).unsqueeze(1)

        # Training
        if not infer:
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long() # Sample a batch of t from U[0,T]
            x_latent = ret['latent']
            ret['diff_loss'] = self.p_losses(x_latent, t)
        
        # Inference
        #   1) Diffuse Ref.Style for :start_from: steps and reconstruct
        #   2) Generate new style reconstructing from a Gaussian 
        else:
            x_latent = ret['latent']
            if isinstance(reconstruct_from, int):
                t = reconstruct_from
                x = self.q_sample(x_start=x_latent, t=torch.tensor([t - 1], device=device).long())
            elif isinstance(reconstruct_from, str) and reconstruct_from == 'gaussian':
                t = self.num_timesteps
                x = torch.randn(x_latent.shape, device=device)
            else:
                raise NotImplementedError
    
            # Iterate the Denoiser for reconstruction
            for i in reversed(range(0, t)):
                x = self.p_sample(x, torch.full((b,), i, device=device, dtype=torch.long))
            ret['style_out'] = x
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
    def __init__(self, in_dims, num_residual_layers, residual_channels, dilation_cycle_length):
        super().__init__()
        # Params
        self.num_residual_layers = num_residual_layers
        self.residual_channels = residual_channels
        self.dilation_cycle_length = dilation_cycle_length

        # Input Conv1x1
        self.input_projection = Conv1d(in_dims, self.residual_channels, 1)
        
        # Sinusoidal Step Embedding and Projection
        self.diffusion_embedding = SinusoidalPosEmb(self.residual_channels)
        dim = self.residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            Mish(),
            nn.Linear(dim * 4, dim)
        )
        
        # Residual Layers
        self.residual_layers = nn.ModuleList([
            ResidualBlock(self.residual_channels, 2 ** (i % self.dilation_cycle_length))
            for i in range(self.num_residual_layers)
        ])
        
        # Conv 1x1
        self.skip_projection = Conv1d(self.residual_channels, self.residual_channels, 1)
        
        # Conv 1x1
        self.output_projection = Conv1d(self.residual_channels, in_dims, 1)
        nn.init.zeros_(self.output_projection.weight)

    def forward(self, specs, diffusion_step):
        """
        :param spec: [B, M, T]
        :param diffusion_step: [B]
        :return:
        """
        # Conv1x1 + ReLU
        x = self.input_projection(specs)  #  [B, residual_channel, T]
        x = F.relu(x)

        # Step Embedding
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step) # [B, 1, residual_channel]

        # Residual Pass
        skip = []
        for layer_id, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, diffusion_step)
            skip.append(skip_connection)

        # Sum Skips
        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        
        # Reultant Skip Project and ReLU
        x = self.skip_projection(x)
        x = F.relu(x)
        
        # Project to Output
        x = self.output_projection(x)  # [B, M, T]
        return x

class ResidualBlock(nn.Module):
    def __init__(self, residual_channels, dilation):
        super().__init__()

        # Input Projection
        self.diffusion_projection = nn.Linear(residual_channels, residual_channels)
        
        # Main Dilated 3x3 Conv
        self.dilated_conv = Conv1d(residual_channels, 2 * residual_channels, 3, padding=dilation, dilation=dilation)
        
        # Output Projection
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

    def forward(self, x, diffusion_step):
        # Project Step Embedding
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1) # [B, 1, residual_channels, 1]

        # Sum Input with Step Embedding
        y = x + diffusion_step
        
        # Perform 3x3 Dilated Conv
        y = self.dilated_conv(y)

        # Split
        gate, filter = torch.chunk(y, 2, dim=1)

        # Dot Product
        y = torch.sigmoid(gate) * torch.tanh(filter)

        # Project Output
        y = self.output_projection(y)
        
        # Split
        residual, skip = torch.chunk(y, 2, dim=1)

        # Residual Out
        residual_out = (x + residual) / sqrt(2.0)
        return residual_out, skip

class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))

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

def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer

@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


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

def main():

    """
    #Variance Schedule Test
    import matplotlib.pyplot as plt
    timesteps = 1000
    plt.plot(np.arange(0, timesteps, 1)/timesteps, np.cumprod(1-cosine_beta_schedule(timesteps)))
    plt.savefig('fig.png')
    plt.plot(np.arange(0, timesteps, 1)/timesteps, np.cumprod(1-linear_beta_schedule(timesteps)))
    plt.savefig('fig2.png')
    
    # Foward Test
    ref_mels = torch.randn((32,512,80))
    a = DiffStyleEncoder()
    print(a.forward(ref_mels, infer = False, reconstruct_from = None))
    print(a.forward(ref_mels, infer = True, reconstruct_from = a.K_step))
    """

if __name__ == '__main__':
    main()

