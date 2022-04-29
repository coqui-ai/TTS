from turtle import forward
from coqpit import Coqpit
from matplotlib.cbook import flatten
from torch import nn
from TTS.utils.audio import TorchSTFT
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.nn import functional as F
from TTS.model import BaseTrainerModel
from TTS.tts.layers.generic.wavenet import WNBlocks
from TTS.tts.utils.visual import plot_spectrogram
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union
from TTS.enhancer.datasets.dataset import EnhancerDataset
import torch.distributed as dist
from librosa.core import resample
from torch.utils.data import DataLoader, Sampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler
import torch
from torch.nn import Upsample
from TTS.enhancer.layers.losses import BWEDiscriminatorLoss, BWEGeneratorLoss
import librosa
from TTS.vocoder.models.melgan_multiscale_discriminator import MelganMultiscaleDiscriminator

@dataclass
class BWEArgs(Coqpit):
    num_channel_wn: int = 128
    dilation_rate_wn: int = 3
    num_blocks_wn: int = 2
    num_layers_wn: int = 7
    kernel_size_wn: int = 3

class ConvLayerSpecDisc(nn.Module):
    def __init__(self, kernel_size, stride, first_layer):
        super().__init__()
        get_padding = lambda k, d: int((k * d - d) / 2)
        self.conv = nn.Conv2d(
            1 if first_layer else 32, 32,
            kernel_size,
            stride,
            padding=(get_padding(kernel_size[0], 1), get_padding(kernel_size[1], 1))
        )
        self.bn = nn.BatchNorm2d(32)

    def forward(self, x):
        x = self.conv(x)
        x = F.leaky_relu(x)
        x = self.bn(x)
        return x
        
class SpectralDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.mel_spec = TorchSTFT(
                            n_fft=1024, 
                            hop_length=512, 
                            win_length=1024, 
                            sample_rate=48000, 
                            n_mels=128,
                            use_mel=True,
                            device="cuda",
                            do_amp_to_db=True)
        self.kernel_sizes = [(7,7), (5,5), (5,5), (5,5)]
        self.layers = nn.ModuleList([
            ConvLayerSpecDisc(
                self.kernel_sizes[i],
                (1,2),
                i == 0
            ) for i in range(len(self.kernel_sizes))
        ])
        self.last_conv = nn.Conv2d(32, 1, (15,5), (15,1))

    def forward(self, x):
        feats = []
        x = self.mel_spec(x).transpose(1,2).unsqueeze(1)
        for disc in self.layers:
            x = disc(x)
            feats.append(x)
        x = self.last_conv(x)
        feats.append(x)
        x = torch.flatten(x, 1, -1)
        return x, feats

class BWE(BaseTrainerModel):
    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor",
    ):
        super().__init__()
        self.config = config
        self.ap = ap
        self.args = config.model_args
        self.input_sr = config.input_sr
        self.target_sr = config.target_sr
        self.scale_factor = (self.target_sr / self.input_sr, )
        self.train_disc = False

        self.postconv = nn.Conv1d(self.args.num_channel_wn, 1, kernel_size=1)
        self.generator = WNBlocks(
            in_channels=1,
            hidden_channels=self.args.num_channel_wn,
            kernel_size=self.args.kernel_size_wn,
            dilation_rate=self.args.dilation_rate_wn,
            num_blocks=self.args.num_blocks_wn,
            num_layers=self.args.num_layers_wn,
        )
        self.waveform_disc = MelganMultiscaleDiscriminator(
            downsample_factors=(2, 2, 2),
            base_channels=16,
            max_channels=1024,
        )
        self.spectral_disc = SpectralDiscriminator()

    def init_from_config(config: Coqpit):
        from TTS.utils.audio import AudioProcessor
        ap = AudioProcessor.init_from_config(config)
        return BWE(config, ap)

    def upsample(self, x):
        device = next(self.parameters()).device
        x = x[:,0,:].cpu().detach().numpy()
        x = resample(x, self.input_sr, self.target_sr, res_type="kaiser_best")
        return torch.tensor(x, device=device, requires_grad=True).unsqueeze(1)

    def gen_forward(self, x):
        x = self.upsample(x)
        x = self.generator(x)
        x = self.postconv(x)
        return {
            "y_hat": x,
        }

    def disc_forward(self, x):
        scores, feats = self.waveform_disc(x)
        score, feats_ = self.spectral_disc(x)
        scores.append(score)
        return scores, feats + feats_

    def forward(self, x):
        return self.gen_forward(x)

    def inference(self, x):
        x = x.unsqueeze(1)
        x = self.gen_forward(x)
        return x

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        outputs = {}
        loss_dict = {}
        x = batch["input_wav"].unsqueeze(1)
        y = batch["target_wav"].unsqueeze(1)
        lens = batch["target_lens"]

        if optimizer_idx not in [0, 1]:
            raise ValueError(" [!] Unexpected `optimizer_idx`.")

        if optimizer_idx == 0:
            y_hat = self.gen_forward(x)["y_hat"]

            self.y_hat_g = y_hat
            scores_fake, feats_fake, feats_real = None, None, None

            if self.train_disc:
                y_d = y.clone()

                scores_fake, feats_fake = self.disc_forward(y_hat.detach())
                scores_real, feats_real = self.disc_forward(y_d)

                loss_dict = criterion[optimizer_idx](scores_fake, scores_real)
                outputs = {"model_outputs": y_hat}

        if optimizer_idx == 1:
            if self.train_disc:
                scores_fake, feats_fake = self.disc_forward(self.y_hat_g)
                with torch.no_grad():
                    _, feats_real = self.disc_forward(y)

            loss_dict = criterion[optimizer_idx](self.y_hat_g, y, lens, scores_fake, feats_fake, feats_real)
            outputs = {"model_outputs": self.y_hat_g}

        return outputs, loss_dict

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        self.train_disc = True # Avoid a bug in the Training with the missing discriminator loss
        out = self.train_step(batch, criterion, optimizer_idx)
        return out

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = EnhancerDataset(
                config,
                self.ap,
                samples,
                augmentation_config=config.audio_augmentation,
                use_torch_spec=None,
                verbose=True,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=True,  # shuffle is done in the dataset.
                drop_last=False,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"], strict=strict)
        if eval:
            self.eval()
            assert not self.training

    def get_optimizer(self) -> List:
        disc_params = list(self.waveform_disc.parameters()) + list(self.spectral_disc.parameters())
        optimizer_disc = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, parameters=disc_params)
        gen_params = list(self.generator.parameters()) + list(self.postconv.parameters())
        optimizer_gen = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_params)
        return [optimizer_disc, optimizer_gen]

    def get_lr(self) -> List:
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> List:
        disc_scheduler = get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)
        gen_scheduler = get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)
        return [disc_scheduler, gen_scheduler]

    def get_criterion(self):
        #device = next(self.parameters()).device
        return [BWEDiscriminatorLoss(), BWEGeneratorLoss("cuda:0")]

    def get_sampler(self, config: Coqpit, dataset: EnhancerDataset, num_gpus=1):
        sampler = None
        # sampler for DDP
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            sampler = DistributedSamplerWrapper(sampler) if num_gpus > 1 else sampler
        return sampler

    def eval_log(
        self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int  # pylint: disable=unused-argument
    ) -> Tuple[Dict, np.ndarray]:
        """Call `_log()` for evaluation."""
        figures, audios = self._log("eval", batch, outputs)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.target_sr)

    def _log(self, name: str, batch: Dict, outputs: Dict) -> Tuple[Dict, Dict]:
        y_hat = outputs[0]["y_hat"][0].detach().squeeze(0).cpu().numpy()
        y = batch["target_wav"][0].detach().squeeze(0).cpu().numpy()
        x = batch["input_wav"][0].detach().squeeze(0).cpu().numpy()
        x = resample(x, 16000, 48000, res_type="kaiser_best")
        figures = {
            name + "_input": self._plot_spec(x, 48000),
            name + "_generated": self._plot_spec(y_hat, 48000),
            name + "_target": self._plot_spec(y, 48000),
        }
        audios = {
            f"{name}_input/audio": x,
            f"{name}_generated/audio": y_hat,
            f"{name}_target/audio": y
        }
        return figures, audios

    @staticmethod
    def _plot_spec(x, sr):
        spec = librosa.feature.melspectrogram(x, sr=sr, n_mels=128, fmax=24000)
        spec = librosa.power_to_db(spec, ref=np.max)
        fig = plt.figure(figsize=(16, 10))
        plt.imshow(spec, aspect="auto", origin="lower")
        plt.colorbar()
        plt.tight_layout()
        plt.close()
        return fig

    def on_train_step_start(self, trainer) -> None:
        """Enable the discriminator training based on `steps_to_start_discriminator`

        Args:
            trainer (Trainer): Trainer object.
        """
        self.train_disc = trainer.total_steps_done >= self.config.steps_to_start_discriminator
