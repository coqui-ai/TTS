from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.nn.utils.parametrizations import weight_norm
from torch.nn.utils.parametrize import remove_parametrizations
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.utils.io import load_fsspec
from TTS.vocoder.datasets import WaveGradDataset
from TTS.vocoder.layers.wavegrad import Conv1d, DBlock, FiLM, UBlock
from TTS.vocoder.models.base_vocoder import BaseVocoder
from TTS.vocoder.utils.generic_utils import plot_results


@dataclass
class WavegradArgs(Coqpit):
    in_channels: int = 80
    out_channels: int = 1
    use_weight_norm: bool = False
    y_conv_channels: int = 32
    x_conv_channels: int = 768
    dblock_out_channels: List[int] = field(default_factory=lambda: [128, 128, 256, 512])
    ublock_out_channels: List[int] = field(default_factory=lambda: [512, 512, 256, 128, 128])
    upsample_factors: List[int] = field(default_factory=lambda: [4, 4, 4, 2, 2])
    upsample_dilations: List[List[int]] = field(
        default_factory=lambda: [[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]]
    )


class Wavegrad(BaseVocoder):
    """🐸 🌊 WaveGrad 🌊 model.
    Paper - https://arxiv.org/abs/2009.00713

    Examples:
        Initializing the model.

        >>> from TTS.vocoder.configs import WavegradConfig
        >>> config = WavegradConfig()
        >>> model = Wavegrad(config)

    Paper Abstract:
        This paper introduces WaveGrad, a conditional model for waveform generation which estimates gradients of the
        data density. The model is built on prior work on score matching and diffusion probabilistic models. It starts
        from a Gaussian white noise signal and iteratively refines the signal via a gradient-based sampler conditioned
        on the mel-spectrogram. WaveGrad offers a natural way to trade inference speed for sample quality by adjusting
        the number of refinement steps, and bridges the gap between non-autoregressive and autoregressive models in
        terms of audio quality. We find that it can generate high fidelity audio samples using as few as six iterations.
        Experiments reveal WaveGrad to generate high fidelity audio, outperforming adversarial non-autoregressive
        baselines and matching a strong likelihood-based autoregressive baseline using fewer sequential operations.
        Audio samples are available at this https URL.
    """

    # pylint: disable=dangerous-default-value
    def __init__(self, config: Coqpit):
        super().__init__(config)
        self.config = config
        self.use_weight_norm = config.model_params.use_weight_norm
        self.hop_len = np.prod(config.model_params.upsample_factors)
        self.noise_level = None
        self.num_steps = None
        self.beta = None
        self.alpha = None
        self.alpha_hat = None
        self.c1 = None
        self.c2 = None
        self.sigma = None

        # dblocks
        self.y_conv = Conv1d(1, config.model_params.y_conv_channels, 5, padding=2)
        self.dblocks = nn.ModuleList([])
        ic = config.model_params.y_conv_channels
        for oc, df in zip(config.model_params.dblock_out_channels, reversed(config.model_params.upsample_factors)):
            self.dblocks.append(DBlock(ic, oc, df))
            ic = oc

        # film
        self.film = nn.ModuleList([])
        ic = config.model_params.y_conv_channels
        for oc in reversed(config.model_params.ublock_out_channels):
            self.film.append(FiLM(ic, oc))
            ic = oc

        # ublocksn
        self.ublocks = nn.ModuleList([])
        ic = config.model_params.x_conv_channels
        for oc, uf, ud in zip(
            config.model_params.ublock_out_channels,
            config.model_params.upsample_factors,
            config.model_params.upsample_dilations,
        ):
            self.ublocks.append(UBlock(ic, oc, uf, ud))
            ic = oc

        self.x_conv = Conv1d(config.model_params.in_channels, config.model_params.x_conv_channels, 3, padding=1)
        self.out_conv = Conv1d(oc, config.model_params.out_channels, 3, padding=1)

        if config.model_params.use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, spectrogram, noise_scale):
        shift_and_scale = []

        x = self.y_conv(x)
        shift_and_scale.append(self.film[0](x, noise_scale))

        for film, layer in zip(self.film[1:], self.dblocks):
            x = layer(x)
            shift_and_scale.append(film(x, noise_scale))

        x = self.x_conv(spectrogram)
        for layer, (film_shift, film_scale) in zip(self.ublocks, reversed(shift_and_scale)):
            x = layer(x, film_shift, film_scale)
        x = self.out_conv(x)
        return x

    def load_noise_schedule(self, path):
        beta = np.load(path, allow_pickle=True).item()["beta"]  # pylint: disable=unexpected-keyword-arg
        self.compute_noise_level(beta)

    @torch.no_grad()
    def inference(self, x, y_n=None):
        """
        Shapes:
            x: :math:`[B, C , T]`
            y_n: :math:`[B, 1, T]`
        """
        if y_n is None:
            y_n = torch.randn(x.shape[0], 1, self.hop_len * x.shape[-1])
        else:
            y_n = torch.FloatTensor(y_n).unsqueeze(0).unsqueeze(0)
        y_n = y_n.type_as(x)
        sqrt_alpha_hat = self.noise_level.to(x)
        for n in range(len(self.alpha) - 1, -1, -1):
            y_n = self.c1[n] * (y_n - self.c2[n] * self.forward(y_n, x, sqrt_alpha_hat[n].repeat(x.shape[0])))
            if n > 0:
                z = torch.randn_like(y_n)
                y_n += self.sigma[n - 1] * z
            y_n.clamp_(-1.0, 1.0)
        return y_n

    def compute_y_n(self, y_0):
        """Compute noisy audio based on noise schedule"""
        self.noise_level = self.noise_level.to(y_0)
        if len(y_0.shape) == 3:
            y_0 = y_0.squeeze(1)
        s = torch.randint(0, self.num_steps - 1, [y_0.shape[0]])
        l_a, l_b = self.noise_level[s], self.noise_level[s + 1]
        noise_scale = l_a + torch.rand(y_0.shape[0]).to(y_0) * (l_b - l_a)
        noise_scale = noise_scale.unsqueeze(1)
        noise = torch.randn_like(y_0)
        noisy_audio = noise_scale * y_0 + (1.0 - noise_scale**2) ** 0.5 * noise
        return noise.unsqueeze(1), noisy_audio.unsqueeze(1), noise_scale[:, 0]

    def compute_noise_level(self, beta):
        """Compute noise schedule parameters"""
        self.num_steps = len(beta)
        alpha = 1 - beta
        alpha_hat = np.cumprod(alpha)
        noise_level = np.concatenate([[1.0], alpha_hat**0.5], axis=0)
        noise_level = alpha_hat**0.5

        # pylint: disable=not-callable
        self.beta = torch.tensor(beta.astype(np.float32))
        self.alpha = torch.tensor(alpha.astype(np.float32))
        self.alpha_hat = torch.tensor(alpha_hat.astype(np.float32))
        self.noise_level = torch.tensor(noise_level.astype(np.float32))

        self.c1 = 1 / self.alpha**0.5
        self.c2 = (1 - self.alpha) / (1 - self.alpha_hat) ** 0.5
        self.sigma = ((1.0 - self.alpha_hat[:-1]) / (1.0 - self.alpha_hat[1:]) * self.beta[1:]) ** 0.5

    def remove_weight_norm(self):
        for _, layer in enumerate(self.dblocks):
            if len(layer.state_dict()) != 0:
                try:
                    remove_parametrizations(layer, "weight")
                except ValueError:
                    layer.remove_weight_norm()

        for _, layer in enumerate(self.film):
            if len(layer.state_dict()) != 0:
                try:
                    remove_parametrizations(layer, "weight")
                except ValueError:
                    layer.remove_weight_norm()

        for _, layer in enumerate(self.ublocks):
            if len(layer.state_dict()) != 0:
                try:
                    remove_parametrizations(layer, "weight")
                except ValueError:
                    layer.remove_weight_norm()

        remove_parametrizations(self.x_conv, "weight")
        remove_parametrizations(self.out_conv, "weight")
        remove_parametrizations(self.y_conv, "weight")

    def apply_weight_norm(self):
        for _, layer in enumerate(self.dblocks):
            if len(layer.state_dict()) != 0:
                layer.apply_weight_norm()

        for _, layer in enumerate(self.film):
            if len(layer.state_dict()) != 0:
                layer.apply_weight_norm()

        for _, layer in enumerate(self.ublocks):
            if len(layer.state_dict()) != 0:
                layer.apply_weight_norm()

        self.x_conv = weight_norm(self.x_conv)
        self.out_conv = weight_norm(self.out_conv)
        self.y_conv = weight_norm(self.y_conv)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
            if self.config.model_params.use_weight_norm:
                self.remove_weight_norm()
            betas = np.linspace(
                config["test_noise_schedule"]["min_val"],
                config["test_noise_schedule"]["max_val"],
                config["test_noise_schedule"]["num_steps"],
            )
            self.compute_noise_level(betas)
        else:
            betas = np.linspace(
                config["train_noise_schedule"]["min_val"],
                config["train_noise_schedule"]["max_val"],
                config["train_noise_schedule"]["num_steps"],
            )
            self.compute_noise_level(betas)

    def train_step(self, batch: Dict, criterion: Dict) -> Tuple[Dict, Dict]:
        # format data
        x = batch["input"]
        y = batch["waveform"]

        # set noise scale
        noise, x_noisy, noise_scale = self.compute_y_n(y)

        # forward pass
        noise_hat = self.forward(x_noisy, x, noise_scale)

        # compute losses
        loss = criterion(noise, noise_hat)
        return {"model_output": noise_hat}, {"loss": loss}

    def train_log(  # pylint: disable=no-self-use
        self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int  # pylint: disable=unused-argument
    ) -> Tuple[Dict, np.ndarray]:
        pass

    @torch.no_grad()
    def eval_step(self, batch: Dict, criterion: nn.Module) -> Tuple[Dict, Dict]:
        return self.train_step(batch, criterion)

    def eval_log(  # pylint: disable=no-self-use
        self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        pass

    def test(self, assets: Dict, test_loader: "DataLoader", outputs=None):  # pylint: disable=unused-argument
        # setup noise schedule and inference
        ap = assets["audio_processor"]
        noise_schedule = self.config["test_noise_schedule"]
        betas = np.linspace(noise_schedule["min_val"], noise_schedule["max_val"], noise_schedule["num_steps"])
        self.compute_noise_level(betas)
        samples = test_loader.dataset.load_test_samples(1)
        for sample in samples:
            x = sample[0]
            x = x[None, :, :].to(next(self.parameters()).device)
            y = sample[1]
            y = y[None, :]
            # compute voice
            y_pred = self.inference(x)
            # compute spectrograms
            figures = plot_results(y_pred, y, ap, "test")
            # Sample audio
            sample_voice = y_pred[0].squeeze(0).detach().cpu().numpy()
        return figures, {"test/audio": sample_voice}

    def get_optimizer(self):
        return get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, self)

    def get_scheduler(self, optimizer):
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    @staticmethod
    def get_criterion():
        return torch.nn.L1Loss()

    @staticmethod
    def format_batch(batch: Dict) -> Dict:
        # return a whole audio segment
        m, y = batch[0], batch[1]
        y = y.unsqueeze(1)
        return {"input": m, "waveform": y}

    def get_data_loader(self, config: Coqpit, assets: Dict, is_eval: True, samples: List, verbose: bool, num_gpus: int):
        ap = assets["audio_processor"]
        dataset = WaveGradDataset(
            ap=ap,
            items=samples,
            seq_len=self.config.seq_len,
            hop_len=ap.hop_length,
            pad_short=self.config.pad_short,
            conv_pad=self.config.conv_pad,
            is_training=not is_eval,
            return_segments=True,
            use_noise_augment=False,
            use_cache=config.use_cache,
            verbose=verbose,
        )
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=num_gpus <= 1,
            drop_last=False,
            sampler=sampler,
            num_workers=self.config.num_eval_loader_workers if is_eval else self.config.num_loader_workers,
            pin_memory=False,
        )
        return loader

    def on_epoch_start(self, trainer):  # pylint: disable=unused-argument
        noise_schedule = self.config["train_noise_schedule"]
        betas = np.linspace(noise_schedule["min_val"], noise_schedule["max_val"], noise_schedule["num_steps"])
        self.compute_noise_level(betas)

    @staticmethod
    def init_from_config(config: "WavegradConfig"):
        return Wavegrad(config)
