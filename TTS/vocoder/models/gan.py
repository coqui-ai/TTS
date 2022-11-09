from inspect import signature
from typing import Dict, List, Tuple

import numpy as np
import torch
from coqpit import Coqpit
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_fsspec
from TTS.vocoder.datasets.gan_dataset import GANDataset
from TTS.vocoder.layers.losses import DiscriminatorLoss, GeneratorLoss
from TTS.vocoder.models import setup_discriminator, setup_generator
from TTS.vocoder.models.base_vocoder import BaseVocoder
from TTS.vocoder.utils.generic_utils import plot_results


class GAN(BaseVocoder):
    def __init__(self, config: Coqpit, ap: AudioProcessor = None):
        """Wrap a generator and a discriminator network. It provides a compatible interface for the trainer.
        It also helps mixing and matching different generator and disciminator networks easily.

        To implement a new GAN models, you just need to define the generator and the discriminator networks, the rest
        is handled by the `GAN` class.

        Args:
            config (Coqpit): Model configuration.
            ap (AudioProcessor): 🐸TTS AudioProcessor instance. Defaults to None.

        Examples:
            Initializing the GAN model with HifiGAN generator and discriminator.
            >>> from TTS.vocoder.configs import HifiganConfig
            >>> config = HifiganConfig()
            >>> model = GAN(config)
        """
        super().__init__(config)
        self.config = config
        self.model_g = setup_generator(config)
        self.model_d = setup_discriminator(config)
        self.train_disc = False  # if False, train only the generator.
        self.y_hat_g = None  # the last generator prediction to be passed onto the discriminator
        self.ap = ap

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the generator's forward pass.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: output of the GAN generator network.
        """
        return self.model_g.forward(x)

    def inference(self, x: torch.Tensor) -> torch.Tensor:
        """Run the generator's inference pass.

        Args:
            x (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: output of the GAN generator network.
        """
        return self.model_g.inference(x)

    def train_step(self, batch: Dict, criterion: Dict, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Compute model outputs and the loss values. `optimizer_idx` selects the generator or the discriminator for
        network on the current pass.

        Args:
            batch (Dict): Batch of samples returned by the dataloader.
            criterion (Dict): Criterion used to compute the losses.
            optimizer_idx (int): ID of the optimizer in use on the current pass.

        Raises:
            ValueError: `optimizer_idx` is an unexpected value.

        Returns:
            Tuple[Dict, Dict]: model outputs and the computed loss values.
        """
        outputs = {}
        loss_dict = {}

        x = batch["input"]
        y = batch["waveform"]

        if optimizer_idx not in [0, 1]:
            raise ValueError(" [!] Unexpected `optimizer_idx`.")

        if optimizer_idx == 0:
            # DISCRIMINATOR optimization

            # generator pass
            y_hat = self.model_g(x)[:, :, : y.size(2)]

            # cache for generator loss
            # pylint: disable=W0201
            self.y_hat_g = y_hat
            self.y_hat_sub = None
            self.y_sub_g = None

            # PQMF formatting
            if y_hat.shape[1] > 1:
                self.y_hat_sub = y_hat
                y_hat = self.model_g.pqmf_synthesis(y_hat)
                self.y_hat_g = y_hat  # save for generator loss
                self.y_sub_g = self.model_g.pqmf_analysis(y)

            scores_fake, feats_fake, feats_real = None, None, None

            if self.train_disc:
                # use different samples for G and D trainings
                if self.config.diff_samples_for_G_and_D:
                    x_d = batch["input_disc"]
                    y_d = batch["waveform_disc"]
                    # use a different sample than generator
                    with torch.no_grad():
                        y_hat = self.model_g(x_d)

                    # PQMF formatting
                    if y_hat.shape[1] > 1:
                        y_hat = self.model_g.pqmf_synthesis(y_hat)
                else:
                    # use the same samples as generator
                    x_d = x.clone()
                    y_d = y.clone()
                    y_hat = self.y_hat_g

                # run D with or without cond. features
                if len(signature(self.model_d.forward).parameters) == 2:
                    D_out_fake = self.model_d(y_hat.detach().clone(), x_d)
                    D_out_real = self.model_d(y_d, x_d)
                else:
                    D_out_fake = self.model_d(y_hat.detach())
                    D_out_real = self.model_d(y_d)

                # format D outputs
                if isinstance(D_out_fake, tuple):
                    # self.model_d returns scores and features
                    scores_fake, feats_fake = D_out_fake
                    if D_out_real is None:
                        scores_real, feats_real = None, None
                    else:
                        scores_real, feats_real = D_out_real
                else:
                    # model D returns only scores
                    scores_fake = D_out_fake
                    scores_real = D_out_real

                # compute losses
                loss_dict = criterion[optimizer_idx](scores_fake, scores_real)
                outputs = {"model_outputs": y_hat}

        if optimizer_idx == 1:
            # GENERATOR loss
            scores_fake, feats_fake, feats_real = None, None, None
            if self.train_disc:
                if len(signature(self.model_d.forward).parameters) == 2:
                    D_out_fake = self.model_d(self.y_hat_g, x)
                else:
                    D_out_fake = self.model_d(self.y_hat_g)
                D_out_real = None

                if self.config.use_feat_match_loss:
                    with torch.no_grad():
                        D_out_real = self.model_d(y)

                # format D outputs
                if isinstance(D_out_fake, tuple):
                    scores_fake, feats_fake = D_out_fake
                    if D_out_real is None:
                        feats_real = None
                    else:
                        _, feats_real = D_out_real
                else:
                    scores_fake = D_out_fake
                    feats_fake, feats_real = None, None

            # compute losses
            loss_dict = criterion[optimizer_idx](
                self.y_hat_g, y, scores_fake, feats_fake, feats_real, self.y_hat_sub, self.y_sub_g
            )
            outputs = {"model_outputs": self.y_hat_g}
        return outputs, loss_dict

    def _log(self, name: str, ap: AudioProcessor, batch: Dict, outputs: Dict) -> Tuple[Dict, Dict]:
        """Logging shared by the training and evaluation.

        Args:
            name (str): Name of the run. `train` or `eval`,
            ap (AudioProcessor): Audio processor used in training.
            batch (Dict): Batch used in the last train/eval step.
            outputs (Dict): Model outputs from the last train/eval step.

        Returns:
            Tuple[Dict, Dict]: log figures and audio samples.
        """
        y_hat = outputs[0]["model_outputs"] if self.train_disc else outputs[1]["model_outputs"]
        y = batch["waveform"]
        figures = plot_results(y_hat, y, ap, name)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name}/audio": sample_voice}
        return figures, audios

    def train_log(
        self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int  # pylint: disable=unused-argument
    ) -> Tuple[Dict, np.ndarray]:
        """Call `_log()` for training."""
        figures, audios = self._log("eval", self.ap, batch, outputs)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    @torch.no_grad()
    def eval_step(self, batch: Dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Call `train_step()` with `no_grad()`"""
        self.train_disc = True  # Avoid a bug in the Training with the missing discriminator loss
        return self.train_step(batch, criterion, optimizer_idx)

    def eval_log(
        self, batch: Dict, outputs: Dict, logger: "Logger", assets: Dict, steps: int  # pylint: disable=unused-argument
    ) -> Tuple[Dict, np.ndarray]:
        """Call `_log()` for evaluation."""
        figures, audios = self._log("eval", self.ap, batch, outputs)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def load_checkpoint(
        self,
        config: Coqpit,
        checkpoint_path: str,
        eval: bool = False,  # pylint: disable=unused-argument, redefined-builtin
        cache: bool = False,
    ) -> None:
        """Load a GAN checkpoint and initialize model parameters.

        Args:
            config (Coqpit): Model config.
            checkpoint_path (str): Checkpoint file path.
            eval (bool, optional): If true, load the model for inference. If falseDefaults to False.
        """
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        # band-aid for older than v0.0.15 GAN models
        if "model_disc" in state:
            self.model_g.load_checkpoint(config, checkpoint_path, eval)
        else:
            self.load_state_dict(state["model"])
            if eval:
                self.model_d = None
                if hasattr(self.model_g, "remove_weight_norm"):
                    self.model_g.remove_weight_norm()

    def on_train_step_start(self, trainer) -> None:
        """Enable the discriminator training based on `steps_to_start_discriminator`

        Args:
            trainer (Trainer): Trainer object.
        """
        self.train_disc = trainer.total_steps_done >= self.config.steps_to_start_discriminator

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.

        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.

        Returns:
            List: optimizers.
        """
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, self.model_g
        )
        optimizer2 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.model_d
        )
        return [optimizer2, optimizer1]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler1 = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler2 = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler2, scheduler1]

    @staticmethod
    def format_batch(batch: List) -> Dict:
        """Format the batch for training.

        Args:
            batch (List): Batch out of the dataloader.

        Returns:
            Dict: formatted model inputs.
        """
        if isinstance(batch[0], list):
            x_G, y_G = batch[0]
            x_D, y_D = batch[1]
            return {"input": x_G, "waveform": y_G, "input_disc": x_D, "waveform_disc": y_D}
        x, y = batch
        return {"input": x, "waveform": y}

    def get_data_loader(  # pylint: disable=no-self-use, unused-argument
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: True,
        samples: List,
        verbose: bool,
        num_gpus: int,
        rank: int = None,  # pylint: disable=unused-argument
    ):
        """Initiate and return the GAN dataloader.

        Args:
            config (Coqpit): Model config.
            ap (AudioProcessor): Audio processor.
            is_eval (True): Set the dataloader for evaluation if true.
            samples (List): Data samples.
            verbose (bool): Log information if true.
            num_gpus (int): Number of GPUs in use.
            rank (int): Rank of the current GPU. Defaults to None.

        Returns:
            DataLoader: Torch dataloader.
        """
        dataset = GANDataset(
            ap=self.ap,
            items=samples,
            seq_len=config.seq_len,
            hop_len=self.ap.hop_length,
            pad_short=config.pad_short,
            conv_pad=config.conv_pad,
            return_pairs=config.diff_samples_for_G_and_D if "diff_samples_for_G_and_D" in config else False,
            is_training=not is_eval,
            return_segments=not is_eval,
            use_noise_augment=config.use_noise_augment,
            use_cache=config.use_cache,
            verbose=verbose,
        )
        dataset.shuffle_mapping()
        sampler = DistributedSampler(dataset, shuffle=True) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=1 if is_eval else config.batch_size,
            shuffle=num_gpus == 0,
            drop_last=False,
            sampler=sampler,
            num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
            pin_memory=False,
        )
        return loader

    def get_criterion(self):
        """Return criterions for the optimizers"""
        return [DiscriminatorLoss(self.config), GeneratorLoss(self.config)]

    @staticmethod
    def init_from_config(config: Coqpit, verbose=True) -> "GAN":
        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        return GAN(config, ap=ap)
