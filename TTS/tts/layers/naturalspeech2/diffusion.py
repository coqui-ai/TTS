import math
from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm
from einops import rearrange, reduce, repeat
from TTS.tts.layers.naturalspeech2.perciver_resampler import PerceiverResampler
from flash_attn.modules.mha import MHA, ParallelMHA
# def generate_causal_mask(seq_len, device):
#     mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
#     mask = mask.masked_fill(mask == 1, float('-inf')).to(device)
#     return mask
class Diffusion(nn.Module):
    def __init__(
        self,
        max_step: int,
        audio_codec_size: int,
        size_: int,
        pre_attention_query_token: int,
        pre_attention_query_size: int,
        pre_attention_head: int,
        wavenet_kernel_size: int,
        wavenet_dilation: int,
        wavenet_stack: int,
        wavenet_dropout_rate: float,
        wavenet_attention_apply_in_stack: int,
        wavenet_attention_head: int,
        noise_schedule: str,
        scale: float
    ):
        super().__init__()
        self.max_step = max_step
        self.size_ = size_
        self.audio_codec_size = audio_codec_size
        self.scale = scale
        self.min_snr_gamma = 5
        self.denoiser = Denoiser(
            audio_codec_size=audio_codec_size,
            diffusion_size=self.size_,
            encoder_size=self.size_,
            speech_prompter_size=self.size_,
            pre_attention_query_size=pre_attention_query_size,
            pre_attention_query_token=pre_attention_query_token,
            pre_attention_head=pre_attention_head,
            wavenet_kernel_size=wavenet_kernel_size,
            wavenet_dilation=wavenet_dilation,
            wavenet_stack=wavenet_stack,
            wavenet_dropout_rate=wavenet_dropout_rate,
            wavenet_attention_apply_in_stack=wavenet_attention_apply_in_stack,
            wavenet_attention_head=wavenet_attention_head,
        )

        if noise_schedule == "linear":
            self.gamma_schedule = simple_linear_schedule
        elif noise_schedule == "cosine":
            self.gamma_schedule = cosine_schedule
        elif noise_schedule == "sigmoid":
            self.gamma_schedule = sigmoid_schedule

    def forward(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.FloatTensor,
        latents: Optional[torch.Tensor] = None,
    ):
        """
        encodings: [Batch, Enc_d, Audio_ct]
        latents: [Batch, Latent_d, Audio_ct]
        """
        if not latents is None:  # train
            noises = torch.randn_like(latents)

        diffusion_steps = torch.rand(size=(latents.size(0),), device=latents.device)

        gammas = self.gamma_schedule(diffusion_steps)
        alphas, sigmas = self.gamma_to_alpha_sigma(gammas, scale=self.scale)

        noised_latents = latents * alphas[:, None, None] + noises * sigmas[:, None, None]
        diffusion_predictions = self.denoiser(
            latents=noised_latents,
            encodings=encodings,
            lengths=lengths,
            speech_prompts=speech_prompts,
            diffusion_steps=diffusion_steps,
        )

        diffusion_targets = noises * alphas[:, None, None] - latents * sigmas[:, None, None]
        diffusion_starts = latents * alphas[:, None, None] - diffusion_predictions * sigmas[:, None, None]
        
        snr = (alphas * alphas) / (sigmas * sigmas)
        maybe_clipped_snr = snr.clone()
        maybe_clipped_snr.clamp_(max = self.min_snr_gamma)
        loss_weight = maybe_clipped_snr / (snr + 1)
        return diffusion_targets, diffusion_predictions, diffusion_starts, loss_weight

    def ddpm(
        self,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.Tensor,
        ddim_steps: int,
        eps: float = 1e-7,  # minimum at float16 precision
    ):
        steps = self.Get_Sampling_Steps(steps=ddim_steps, references=encodings)

        latents = torch.randn(size=(encodings.size(0), self.audio_codec_size, encodings.size(2)), device=encodings.device)

        for current_steps, next_steps in tqdm(steps):
            gammas = self.gamma_schedule(current_steps)
            alphas, sigmas = self.gamma_to_alpha_sigma(gammas, scale=self.scale)
            log_snrs = self.gamma_to_log_snr(gammas)
            alphas, sigmas, log_snrs = alphas[:, None, None], sigmas[:, None, None], log_snrs[:, None, None]

            next_gammas = self.gamma_schedule(next_steps)
            next_alphas, next_sigmas = self.gamma_to_alpha_sigma(next_gammas, scale=self.scale)
            next_log_snrs = self.gamma_to_log_snr(next_gammas)
            next_alphas, next_sigmas, next_log_snrs = (
                next_alphas[:, None, None],
                next_sigmas[:, None, None],
                next_log_snrs[:, None, None],
            )

            coefficients = -torch.expm1(log_snrs - next_log_snrs)

            noised_predictions = self.denoiser(
                latents=latents,
                encodings=encodings,
                lengths=lengths,
                speech_prompts=speech_prompts,
                diffusion_steps=current_steps,
            )

            epsilons = latents * alphas - noised_predictions * sigmas
            # epsilons.clamp_(-1.0, 1.0)  # clipped

            posterior_means = next_alphas * (latents * (1.0 - coefficients) / alphas + coefficients * epsilons)
            posterior_log_varainces = torch.log(torch.clamp(next_sigmas**2 * coefficients, min=eps))

            noises = torch.randn_like(latents)
            masks = (current_steps > 0).float().unsqueeze(1).unsqueeze(1)  # [Batch, 1, 1]
            latents = posterior_means + masks * (0.5 * posterior_log_varainces).exp() * noises

        return latents

    def ddim(self, encodings: torch.Tensor, lengths: torch.Tensor, speech_prompts: torch.Tensor, ddim_steps: int, scale=1.0):
        steps = self.Get_Sampling_Steps(steps=ddim_steps, references=encodings)

        latents = torch.randn(size=(encodings.size(0), self.audio_codec_size, encodings.size(2)), device=encodings.device)
        for current_steps, next_steps in tqdm(steps):
            gammas = self.gamma_schedule(current_steps)
            alphas, sigmas = self.gamma_to_alpha_sigma(gammas, scale=self.scale)
            alphas, sigmas = alphas[:, None, None], sigmas[:, None, None]

            next_gammas = self.gamma_schedule(next_steps)
            next_alphas, next_sigmas = self.gamma_to_alpha_sigma(next_gammas, scale=self.scale)
            next_alphas, next_sigmas = next_alphas[:, None, None], next_sigmas[:, None, None]

            noised_predictions = self.denoiser(
                latents=latents,
                encodings=encodings,
                lengths=lengths,
                speech_prompts=speech_prompts,
                diffusion_steps=current_steps,
            )

            epsilons = latents * alphas - noised_predictions * sigmas

            noises = (latents - alphas * epsilons) / sigmas
            latents = epsilons * next_alphas + noises * next_sigmas

        return latents

    def Get_Sampling_Steps(self, steps: int, references: torch.Tensor):
        steps = torch.linspace(start=1.0, end=0.0, steps=steps + 1, device=references.device)  # [Step + 1]
        steps = torch.stack([steps[:-1], steps[1:]], dim= 0) # [2, Step]
        steps = steps.unsqueeze(1).expand(-1, references.size(0), -1)    # [2, Batch, Step]
        steps = steps.unbind(dim= 2)
        return steps

    def gamma_to_alpha_sigma(self, gamma, scale=1):
        return torch.sqrt(gamma) * scale, torch.sqrt(1 - gamma)

    def gamma_to_log_snr(self, gamma, scale=1, eps=1e-10):
        return torch.log(torch.clamp(gamma * (scale**2) / (1 - gamma), min=eps))


class Denoiser(torch.nn.Module):
    def __init__(
        self,
        audio_codec_size: int,
        diffusion_size: int,
        encoder_size: int,
        speech_prompter_size: int,
        pre_attention_query_size: int,
        wavenet_kernel_size: int,
        wavenet_dilation: int,
        wavenet_stack: int,
        wavenet_dropout_rate: float,
        wavenet_attention_apply_in_stack: int,
        wavenet_attention_head: int,
        pre_attention_query_token: int = 32,
        pre_attention_head: int = 8,
    ):
        super().__init__()
        self.wavenet_stack = wavenet_stack
        self.pre_attention_query_token = pre_attention_query_token
        self.pre_attention_query_size = pre_attention_query_size
        self.prenet = torch.nn.Sequential(
            torch.nn.Conv1d(in_channels=audio_codec_size, out_channels=diffusion_size, kernel_size=1), torch.nn.ReLU()
        )

        self.step_ffn = DiffusionFFN(diffusion_size=diffusion_size)
        self.pre_attention_query = torch.nn.Parameter(
            torch.empty(1, self.pre_attention_query_token, self.pre_attention_query_size)
            )
        query_variance = math.sqrt(3.0) * math.sqrt(2.0 / (self.pre_attention_query_token + self.pre_attention_query_token))
        self.pre_attention_query.data.uniform_(-query_variance, query_variance)
        # self.pre_attention = nn.MultiheadAttention(diffusion_size, pre_attention_head)
        self.pre_attention = MHA(diffusion_size,
                pre_attention_head,
                num_heads_kv=pre_attention_head,
                cross_attn=True,
                causal=True,
                fused_bias_fc=False,
                use_flash_attn=False)
        # self.perceiver_resampler = PerceiverResampler(
        #         d_model = diffusion_size,
        #         dim_context = audio_codec_size,
        #         num_latents = pre_attention_query_token,
        #         n_heads = pre_attention_head,
        #     )
        self.wavenets = torch.nn.ModuleList(
            [
                WaveNet(
                    channels=diffusion_size,
                    kernel_size=wavenet_kernel_size,
                    dilation=wavenet_dilation,
                    condition_channels=diffusion_size,
                    diffusion_step_channels=diffusion_size,
                    wavenet_dropout_rate=wavenet_dropout_rate,
                    apply_film=(wavenet_index + 1) % wavenet_attention_apply_in_stack == 0,
                    speech_prompt_channels=speech_prompter_size,
                    speech_prompt_attention_head=wavenet_attention_head,
                    block_index=wavenet_index,  # pass the index to track the block number
                )
                for wavenet_index in range(wavenet_stack)
            ]
        )

        self.postnet = torch.nn.Sequential(
            torch.nn.ReLU(), torch.nn.Conv1d(diffusion_size, audio_codec_size, 1)
        )
    def forward(
        self,
        latents: torch.Tensor,
        encodings: torch.Tensor,
        lengths: torch.Tensor,
        speech_prompts: torch.Tensor,
        diffusion_steps: torch.Tensor,
    ):
        """
        latents: [Batch, Codec_d, Audio_ct]
        encodings: [Batch, Enc_d, Audio_ct]
        diffusion_steps: [Batch]
        speech_prompts: [Batch, Prompt_d, Prompt_t]
        """
        masks = (~Mask_Generate(lengths, max_length=latents.size(2))).unsqueeze(1).float()  # [Batch, 1, Audio_ct]
        x = self.prenet(latents)  # [Batch, Diffusion_d, Audio_ct]
        diffusion_steps = self.step_ffn(diffusion_steps)  # [Batch, Diffusion_d, 1]
        speech_prompts=speech_prompts.transpose(1, 2)

        pre_att = self.pre_attention(self.pre_attention_query.expand(speech_prompts.shape[0], -1,  -1), speech_prompts)
        # pre_att=pre_att.permute(1, 2, 0)
        # pre_att = self.perceiver_resampler(speech_prompts)
        # speech_prompts, _ = self.pre_attention(pre_att, speech_prompts, speech_prompts)  # [Batch, Diffusion_d, Token_n]
        # Collect skip connections
        skips_list = []
        encoding_outputs = []
        for wavenet in self.wavenets[:self.wavenet_stack//2]:  # First half of WaveNet blocks (encoding)
            x, skips = wavenet(x, masks, encodings, diffusion_steps, pre_att)
            skips_list.append(skips)
            encoding_outputs.append(x)

        for i, wavenet in enumerate(self.wavenets[self.wavenet_stack//2:]):  # Second half of WaveNet blocks (decoding)
            # Concatenate the corresponding encoding output
            x = x + encoding_outputs[-(i+1)]
            x, skips = wavenet(x, masks, encodings, diffusion_steps, pre_att)
            skips_list.append(skips)
        # skips_list = []
        # for wavenet in self.wavenets:
        #     x, skips = wavenet(
        #         x=x,
        #         masks=masks,
        #         conditions=encodings,
        #         diffusion_steps=diffusion_steps,
        #         speech_prompts=pre_att,
        #     )  # [Batch, Diffusion_d, Audio_ct]
        #     skips_list.append(skips)
        x = torch.stack(skips_list, dim=0).sum(dim=0) / math.sqrt(self.wavenet_stack)
        
        x = self.postnet(x) * masks
        return x

class RMSNorm(nn.Module):
    def __init__(self, dim, scale = True, dim_cond = None):
        super().__init__()
        self.to_gamma_beta = None
        if dim_cond is not None:
            self.to_gamma_beta = nn.Linear(dim_cond, dim * 2) if self.cond else None

        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim)) if scale else None

    def forward(self, x, cond = None):
        out = nn.functional.normalize(x, dim = -1) * self.scale * self.gamma
        
        if self.to_gamma_beta is None:
            return out

        gamma, beta = self.to_gamma_beta(cond).chunk(2, dim = -1)
        gamma, beta = map(lambda t: rearrange(t, 'b d -> b 1 d'), (self.gamma, beta))
        return out * gamma + beta

class DiffusionFFN(nn.Module):
    def __init__(self, diffusion_size):
        super().__init__()

        self.diffusion_embedding = Diffusion_Embedding(channels=diffusion_size)
        self.conv1 = torch.nn.Conv1d(in_channels=diffusion_size + 1, out_channels=diffusion_size * 4, kernel_size=1)
        self.silu = torch.nn.ReLU()
        self.conv2 = torch.nn.Conv1d(in_channels=diffusion_size * 4, out_channels=diffusion_size, kernel_size=1)

    def forward(self, x):
        x = self.diffusion_embedding(x)
        x = x.unsqueeze(2)
        x = self.conv1(x)
        x = self.silu(x)
        x = self.conv2(x)

        return x


class Diffusion_Embedding(torch.nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.weight = torch.nn.Parameter(torch.randn(self.channels // 2))

    def forward(self, x: torch.Tensor):
        x = x.unsqueeze(1)  # [Batch, 1]
        embeddings = x * self.weight.unsqueeze(0) * 2.0 * math.pi  # [Batch, Dim // 2]
        embeddings = torch.cat([x, embeddings.sin(), embeddings.cos()], dim=1)  # [Batch, Dim + 1]

        return embeddings


class WaveNet(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        kernel_size: int,
        dilation: int,
        condition_channels: int,
        diffusion_step_channels: int,
        wavenet_dropout_rate: float = 0.0,
        apply_film: bool = False,
        speech_prompt_channels: Optional[int] = None,
        speech_prompt_attention_head: Optional[int] = None,
        block_index: int = 0,  # add block index to track the block's number in the stack
    ):
        super().__init__()
        self.calc_channels = channels
        self.apply_film = apply_film # apply film every 3rd layer

        self.conv = nn.Conv1d(
            in_channels=channels,
            out_channels=channels * 2,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=(kernel_size - 1) * dilation // 2,
        )

        self.dropout = torch.nn.Dropout(p=wavenet_dropout_rate)

        self.condition = nn.Conv1d(in_channels=condition_channels, out_channels=channels * 2, kernel_size=1)
        self.diffusion_step = nn.Conv1d(in_channels=diffusion_step_channels, out_channels=channels, kernel_size=1)

        if apply_film:
            # self.attention = nn.MultiheadAttention(speech_prompt_channels, speech_prompt_attention_head)
            self.attention = MHA(speech_prompt_channels,
                speech_prompt_attention_head,
                num_heads_kv=speech_prompt_attention_head,
                cross_attn=True,
                dropout=wavenet_dropout_rate,
                causal=True,
                fused_bias_fc=False,
                use_flash_attn=False)
            self.film = FilM(
                channels=channels * 2,
                condition_channels=channels,
            )

    def forward(
        self,
        x: torch.FloatTensor,
        masks: torch.FloatTensor,
        conditions: torch.FloatTensor,
        diffusion_steps: torch.FloatTensor,
        speech_prompts: Optional[torch.FloatTensor],
    ):
        residuals = x
        queries = x = x + self.diffusion_step(diffusion_steps)  # [Batch, Calc_d, Time]
        cond = self.condition(conditions)
        conv = self.conv(x)
        # print(queries.shape, speech_prompts.shape)
        x = conv + cond  # torch.cat([conv,cond],dim=2) # [Batch, Calc_d * 2, Time]
        if self.apply_film:
            # seq_len = queries.permute(2, 0, 1).size(0)
            # attn_mask = generate_causal_mask(seq_len, queries.device)
            prompt_conditions = self.attention(
                queries.transpose(1,2),
                speech_prompts)  # [Batch, Diffusion_d, Time]
            x = self.film(x, prompt_conditions.transpose(1,2), masks)
        x = Fused_Gate(x)  #[Batch, Calc_d, Time]
        x = self.dropout(x) #[Batch, Calc_d, Time]

        return x + residuals, x


def Fused_Gate(x):
    x_tanh, x_sigmoid = x.chunk(chunks=2, dim=1)
    x = x_tanh.tanh() * x_sigmoid.sigmoid()
    return x


class FilM(nn.Conv1d):
    def __init__(
        self,
        channels: int,
        condition_channels: int,
    ):
        super().__init__(in_channels=condition_channels, out_channels=channels * 2, kernel_size=1)

    def forward(self, x: torch.Tensor, conditions: torch.Tensor, masks: torch.Tensor):
        betas, gammas = super().forward(conditions * masks).chunk(2, dim=1)
        x = gammas * x + betas

        return x * masks


def Mask_Generate(lengths: torch.Tensor, max_length: Optional[Union[int, torch.Tensor]] = None):
    """
    lengths: [Batch]
    max_lengths: an int value. If None, max_lengths == max(lengths)
    """
    max_length = max_length or torch.max(lengths)
    sequence = torch.arange(max_length)[None, :].to(lengths.device)
    return sequence >= lengths[:, None]  # [Batch, Time]


def sigmoid_schedule(t, start=-1, end=1, tau=1, clamp_min=1e-9):
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    gamma = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    return gamma.clamp_(min=clamp_min)


def simple_linear_schedule(t, clip_min=1e-9):
    return (1 - t).clamp(min=clip_min)


def cosine_schedule(t, start=0, end=1, tau=1, clip_min=1e-9):
    power = 2 * tau
    v_start = torch.cos(torch.tensor(start) * torch.tensor(math.pi / 2)) ** power
    v_end = torch.cos(torch.tensor(end) * torch.tensor(math.pi / 2)) ** power
    output = torch.cos((t * (end - start) + start) * torch.tensor(math.pi / 2)) ** power
    output = (v_end - output) / (v_end - v_start)
    return output.clamp(min=clip_min)
