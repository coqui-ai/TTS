# -*- coding: utf-8 -*-

import glob
import importlib
import logging
import os
import platform
import re
import sys
import time
import traceback
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
from coqpit import Coqpit
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader

from TTS.config import load_config, register_config
from TTS.tts.datasets import load_meta_data
from TTS.tts.models import setup_model as setup_tts_model
from TTS.tts.utils.text.symbols import parse_symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.callbacks import TrainerCallback
from TTS.utils.distribute import init_distributed
from TTS.utils.generic_utils import (
    KeepAverage,
    count_parameters,
    create_experiment_folder,
    get_git_branch,
    remove_experiment_folder,
    set_init_dict,
    to_cuda,
)
from TTS.utils.io import copy_model_files, save_best_model, save_checkpoint
from TTS.utils.logging import ConsoleLogger, TensorboardLogger
from TTS.utils.trainer_utils import get_optimizer, get_scheduler, is_apex_available, setup_torch_training_env
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data
from TTS.vocoder.models import setup_model as setup_vocoder_model

if platform.system() != "Windows":
    # https://github.com/pytorch/pytorch/issues/973
    import resource

    rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE, (4096, rlimit[1]))

if is_apex_available():
    from apex import amp


@dataclass
class TrainingArgs(Coqpit):
    """Trainer arguments to be defined externally. It helps integrating the `Trainer` with the higher level APIs and
    set the values for distributed training."""

    continue_path: str = field(
        default="",
        metadata={
            "help": "Path to a training folder to continue training. Restore the model from the last checkpoint and continue training under the same folder."
        },
    )
    restore_path: str = field(
        default="",
        metadata={
            "help": "Path to a model checkpoit. Restore the model with the given checkpoint and start a new training."
        },
    )
    best_path: str = field(
        default="",
        metadata={
            "help": "Best model file to be used for extracting best loss. If not specified, the latest best model in continue path is used"
        },
    )
    config_path: str = field(default="", metadata={"help": "Path to the configuration file."})
    rank: int = field(default=0, metadata={"help": "Process rank in distributed training."})
    group_id: str = field(default="", metadata={"help": "Process group id in distributed training."})


class Trainer:
    def __init__(
        self,
        args: Union[Coqpit, Namespace],
        config: Coqpit,
        output_path: str,
        c_logger: ConsoleLogger = None,
        tb_logger: TensorboardLogger = None,
        model: nn.Module = None,
        cudnn_benchmark: bool = False,
    ) -> None:
        """Simple yet powerful 🐸💬 TTS trainer for PyTorch. It can train all the available `tts` and `vocoder` models
        or easily be customized.

        Notes:

            Supports Automatic Mixed Precision training. If `Apex` is availabe, it automatically picks that, else
            it uses PyTorch's native `amp` module. `Apex` may provide more stable training in some cases.

        Args:

            args (Union[Coqpit, Namespace]): Training arguments parsed either from console by `argparse` or `TrainingArgs`
                config object.

            config (Coqpit): Model config object. It includes all the values necessary for initializing, training, evaluating
                and testing the model.

            output_path (str): Path to the output training folder. All the files are saved under thi path.

            c_logger (ConsoleLogger, optional): Console logger for printing training status. If not provided, the default
                console logger is used. Defaults to None.

            tb_logger (TensorboardLogger, optional): Tensorboard logger. If not provided, the default logger is used.
                Defaults to None.

            model (nn.Module, optional): Initialized and ready-to-train model. If it is not defined, `Trainer`
                initializes a model from the provided config. Defaults to None.

            cudnn_benchmark (bool): enable/disable PyTorch cudnn benchmarking. It is better to disable if the model input
                length is changing batch to batch along the training.

        Examples:

            Running trainer on a model.

            >>> args = TrainingArgs(...)
            >>> config = HifiganConfig(...)
            >>> model = GANModel(config)
            >>> trainer = Trainer(args, config, output_path, model=model)
            >>> trainer.fit()

            Running trainer on a config.

            >>> config = WavegradConfig(data_path="/home/erogol/nvme/gdrive/Datasets/LJSpeech-1.1/wavs/", output_path=output_path,)
            >>> args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
            >>> trainer = Trainer(args, config, output_path, c_logger, tb_logger)
            >>> trainer.fit()

        TODO:
            - Accumulate gradients b/w batches.
            - Deepspeed integration
            - Profiler integration.
            - Overfitting to a batch.
            - TPU training
        """

        # set and initialize Pytorch runtime
        self.use_cuda, self.num_gpus = setup_torch_training_env(True, cudnn_benchmark)

        if config is None:
            # parse config from console arguments
            config, output_path, _, c_logger, tb_logger = process_args(args)

        self.output_path = output_path
        self.args = args
        self.config = config

        # init loggers
        self.c_logger = ConsoleLogger() if c_logger is None else c_logger
        if tb_logger is None:
            self.tb_logger = TensorboardLogger(output_path, model_name=config.model)
            self.tb_logger.tb_add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)
        else:
            self.tb_logger = tb_logger
        log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
        self._setup_logger_config(log_file)

        self.total_steps_done = 0
        self.epochs_done = 0
        self.restore_step = 0
        self.best_loss = float("inf")
        self.train_loader = None
        self.eval_loader = None
        self.output_audio_path = os.path.join(output_path, "test_audios")

        self.keep_avg_train = None
        self.keep_avg_eval = None

        self.use_apex = self._is_apex_available()
        self.use_amp_scaler = self.config.mixed_precision and self.use_cuda

        # init audio processor
        self.ap = AudioProcessor(**self.config.audio.to_dict())

        # load dataset samples
        # TODO: refactor this
        if "datasets" in self.config:
            # load data for `tts` models
            self.data_train, self.data_eval = load_meta_data(self.config.datasets)
        elif self.config.feature_path is not None:
            # load data for `vocoder`models
            print(f" > Loading features from: {self.config.feature_path}")
            self.data_eval, self.data_train = load_wav_feat_data(
                self.config.data_path, self.config.feature_path, self.config.eval_split_size
            )
        else:
            # load data for `vocoder`models
            self.data_eval, self.data_train = load_wav_data(self.config.data_path, self.config.eval_split_size)

        # init TTS model
        if model is not None:
            self.model = model
        else:
            self.model = self.get_model(self.config, self.data_train)

        # setup criterion
        self.criterion = self.get_criterion(self.model)

        # DISTRUBUTED
        if self.num_gpus > 1:
            init_distributed(
                args.rank,
                self.num_gpus,
                args.group_id,
                self.config.distributed_backend,
                self.config.distributed_url,
            )

        if self.use_cuda:
            self.model.cuda()
            if isinstance(self.criterion, list):
                self.criterion = [x.cuda() for x in self.criterion]
            else:
                self.criterion.cuda()

        # setup optimizer
        self.optimizer = self.get_optimizer(self.model, self.config)

        # callback
        self.callbacks = TrainerCallback(self)
        self.callbacks.on_init_start()

        # init AMP
        if self.use_amp_scaler:
            if self.use_apex:
                self.scaler = None
                self.model, self.optimizer = amp.initialize(self.model, self.optimizer, opt_level="O1")
            if isinstance(self.optimizer, list):
                self.scaler = [torch.cuda.amp.GradScaler()] * len(self.optimizer)
            else:
                self.scaler = torch.cuda.amp.GradScaler()
        else:
            self.scaler = None

        if self.args.restore_path:
            self.model, self.optimizer, self.scaler, self.restore_step = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer, self.scaler
            )

        # setup scheduler
        self.scheduler = self.get_scheduler(self.model, self.config, self.optimizer)

        # DISTRUBUTED
        if self.num_gpus > 1:
            self.model = DDP_th(self.model, device_ids=[args.rank], output_device=args.rank)

        # count model size
        num_params = count_parameters(self.model)
        print("\n > Model has {} parameters".format(num_params))

        self.callbacks.on_init_end()

    @staticmethod
    def get_model(config: Coqpit, data) -> nn.Module:
        """Initialize model from config.

        Args:
            config (Coqpit): Model config.

        Returns:
            nn.Module: initialized model.
        """
        # TODO: better model setup
        try:
            model = setup_tts_model(config, data)
        except ModuleNotFoundError:
            model = setup_vocoder_model(config)
        return model

    def restore_model(
        self,
        config: Coqpit,
        restore_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]:
        """Restore training from an old run. It restores model, optimizer, AMP scaler and training stats.

        Args:
            config (Coqpit): Model config.
            restore_path (str): Path to the restored training run.
            model (nn.Module): Model to restored.
            optimizer (torch.optim.Optimizer): Optimizer to restore.
            scaler (torch.cuda.amp.GradScaler, optional): AMP scaler to restore. Defaults to None.

        Returns:
            Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]: [description]
        """

        def _restore_list_objs(states, obj):
            if isinstance(obj, list):
                for idx, state in enumerate(states):
                    obj[idx].load_state_dict(state)
            else:
                obj.load_state_dict(states)
            return obj

        print(" > Restoring from %s ..." % os.path.basename(restore_path))
        checkpoint = torch.load(restore_path)
        try:
            print(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            print(" > Restoring Optimizer...")
            optimizer = _restore_list_objs(checkpoint["optimizer"], optimizer)
            if "scaler" in checkpoint and self.use_amp_scaler and checkpoint["scaler"]:
                print(" > Restoring AMP Scaler...")
                scaler = _restore_list_objs(checkpoint["scaler"], scaler)
        except (KeyError, RuntimeError):
            print(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        if isinstance(self.optimizer, list):
            for idx, optim in enumerate(optimizer):
                for group in optim.param_groups:
                    group["lr"] = self.get_lr(model, config)[idx]
        else:
            for group in optimizer.param_groups:
                group["lr"] = self.get_lr(model, config)
        print(
            " > Model restored from step %d" % checkpoint["step"],
        )
        restore_step = checkpoint["step"]
        return model, optimizer, scaler, restore_step

    @staticmethod
    def _get_loader(
        model: nn.Module,
        config: Coqpit,
        ap: AudioProcessor,
        is_eval: bool,
        data_items: List,
        verbose: bool,
        num_gpus: int,
    ) -> DataLoader:
        if hasattr(model, "get_data_loader"):
            loader = model.get_data_loader(config, ap, is_eval, data_items, verbose, num_gpus)
        return loader

    def get_train_dataloader(self, ap: AudioProcessor, data_items: List, verbose: bool) -> DataLoader:
        """Initialize and return a training data loader.

        Args:
            ap (AudioProcessor): Audio processor.
            data_items (List): Data samples used for training.
            verbose (bool): enable/disable printing loader stats at initialization.

        Returns:
            DataLoader: Initialized training data loader.
        """
        return self._get_loader(self.model, self.config, ap, False, data_items, verbose, self.num_gpus)

    def get_eval_dataloader(self, ap: AudioProcessor, data_items: List, verbose: bool) -> DataLoader:
        return self._get_loader(self.model, self.config, ap, True, data_items, verbose, self.num_gpus)

    def format_batch(self, batch: List) -> Dict:
        """Format the dataloader output and return a batch.

        Args:
            batch (List): Batch returned by the dataloader.

        Returns:
            Dict: Formatted batch.
        """
        batch = self.model.format_batch(batch)
        if self.use_cuda:
            for k, v in batch.items():
                batch[k] = to_cuda(v)
        return batch

    @staticmethod
    def _model_train_step(
        batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a trainig forward step. Compute model outputs and losses.

        Args:
            batch (Dict): [description]
            model (nn.Module): [description]
            criterion (nn.Module): [description]
            optimizer_idx (int, optional): [description]. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: [description]
        """
        input_args = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        # unwrap model in DDP training
        if hasattr(model, "module"):
            return model.module.train_step(*input_args)
        return model.train_step(*input_args)

    def _optimize(
        self,
        batch: Dict,
        model: nn.Module,
        optimizer: Union[torch.optim.Optimizer, List],
        scaler: "AMPScaler",
        criterion: nn.Module,
        scheduler: Union[torch.optim.lr_scheduler._LRScheduler, List],  # pylint: disable=protected-access
        config: Coqpit,
        optimizer_idx: int = None,
    ) -> Tuple[Dict, Dict, int, torch.Tensor]:
        """Perform a forward - backward pass and run the optimizer.

        Args:
            batch (Dict): Input batch. If
            model (nn.Module): Model for training. Defaults to None.
            optimizer (Union[nn.optim.Optimizer, List]): Model's optimizer. If it is a list then, `optimizer_idx` must be defined to indicate the optimizer in use.
            scaler (AMPScaler): AMP scaler.
            criterion (nn.Module): Model's criterion.
            scheduler (Union[torch.optim.lr_scheduler._LRScheduler, List]): LR scheduler used by the optimizer.
            config (Coqpit): Model config.
            optimizer_idx (int, optional): Target optimizer being used. Defaults to None.

        Raises:
            RuntimeError: When the loss is NaN.

        Returns:
            Tuple[Dict, Dict, int, torch.Tensor]: model outputs, losses, step time and gradient norm.
        """
        step_start_time = time.time()
        # zero-out optimizer
        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=config.mixed_precision):
            if optimizer_idx is not None:
                outputs, loss_dict = self._model_train_step(batch, model, criterion, optimizer_idx=optimizer_idx)
            else:
                outputs, loss_dict = self._model_train_step(batch, model, criterion)

        # skip the rest
        if outputs is None:
            step_time = time.time() - step_start_time
            return None, {}, step_time, 0

        # check nan loss
        if torch.isnan(loss_dict["loss"]).any():
            raise RuntimeError(f"Detected NaN loss at step {self.total_steps_done}.")

        # set gradient clipping threshold
        if "grad_clip" in config and config.grad_clip is not None:
            if optimizer_idx is not None:
                grad_clip = config.grad_clip[optimizer_idx]
            else:
                grad_clip = config.grad_clip
        else:
            grad_clip = 0.0  # meaning no gradient clipping

        # TODO: compute grad norm
        if grad_clip <= 0:
            grad_norm = 0

        # optimizer step
        update_lr_scheduler = True
        if self.use_amp_scaler:
            if self.use_apex:
                with amp.scale_loss(loss_dict["loss"], optimizer) as scaled_loss:
                    scaled_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer),
                    grad_clip,
                )
            else:
                # model optimizer step in mixed precision mode
                scaler.scale(loss_dict["loss"]).backward()
                scaler.unscale_(optimizer)
                if grad_clip > 0:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scale_prev = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                update_lr_scheduler = scale_prev <= scaler.get_scale()
        else:
            # main model optimizer step
            loss_dict["loss"].backward()
            if grad_clip > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        step_time = time.time() - step_start_time

        # setup lr
        if scheduler is not None and update_lr_scheduler:
            scheduler.step()

        # detach losses
        loss_dict = self._detach_loss_dict(loss_dict)
        if optimizer_idx is not None:
            loss_dict[f"loss_{optimizer_idx}"] = loss_dict.pop("loss")
            loss_dict[f"grad_norm_{optimizer_idx}"] = grad_norm
        return outputs, loss_dict, step_time, grad_norm

    @staticmethod
    def _detach_loss_dict(loss_dict: Dict) -> Dict:
        """Detach loss values from autograp.

        Args:
            loss_dict (Dict): losses.

        Returns:
            Dict: losses detached from autograph.
        """
        loss_dict_detached = {}
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_detached[key] = value
            else:
                loss_dict_detached[key] = value.item()
        return loss_dict_detached

    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        """Perform a training step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            batch_n_steps (int): Number of steps needed to complete an epoch. Needed for logging.
            step (int): Current step number in this epoch.
            loader_start_time (float): The time when the data loading is started. Needed for logging.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        self.callbacks.on_train_step_start()
        # format data
        batch = self.format_batch(batch)
        loader_time = time.time() - loader_start_time

        # conteainers to hold model outputs and losses for each optimizer.
        outputs_per_optimizer = None
        log_dict = {}
        loss_dict = {}
        if not isinstance(self.optimizer, list):
            # training with a single optimizer
            outputs, loss_dict_new, step_time, grad_norm = self._optimize(
                batch, self.model, self.optimizer, self.scaler, self.criterion, self.scheduler, self.config
            )
            loss_dict.update(loss_dict_new)
        else:
            # training with multiple optimizers (e.g. GAN)
            outputs_per_optimizer = [None] * len(self.optimizer)
            total_step_time = 0
            for idx, optimizer in enumerate(self.optimizer):
                criterion = self.criterion
                scaler = self.scaler[idx] if self.use_amp_scaler else None
                scheduler = self.scheduler[idx]
                outputs, loss_dict_new, step_time, grad_norm = self._optimize(
                    batch, self.model, optimizer, scaler, criterion, scheduler, self.config, idx
                )
                # skip the rest if the model returns None
                total_step_time += step_time
                outputs_per_optimizer[idx] = outputs
                # if None, model skipped this optimizer
                if loss_dict_new is not None:
                    loss_dict.update(loss_dict_new)
            outputs = outputs_per_optimizer

        # update avg stats
        keep_avg_update = dict()
        for key, value in log_dict.items():
            keep_avg_update["avg_" + key] = value
        keep_avg_update["avg_loader_time"] = loader_time
        keep_avg_update["avg_step_time"] = step_time
        self.keep_avg_train.update_values(keep_avg_update)

        # print training progress
        if self.total_steps_done % self.config.print_step == 0:
            # log learning rates
            lrs = {}
            if isinstance(self.optimizer, list):
                for idx, optimizer in enumerate(self.optimizer):
                    current_lr = self.optimizer[idx].param_groups[0]["lr"]
                    lrs.update({f"current_lr_{idx}": current_lr})
            else:
                current_lr = self.optimizer.param_groups[0]["lr"]
                lrs = {"current_lr": current_lr}
            log_dict.update(lrs)
            if grad_norm > 0:
                log_dict.update({"grad_norm": grad_norm})
            # log run-time stats
            log_dict.update(
                {
                    "step_time": round(step_time, 4),
                    "loader_time": round(loader_time, 4),
                }
            )
            self.c_logger.print_train_step(
                batch_n_steps, step, self.total_steps_done, log_dict, loss_dict, self.keep_avg_train.avg_values
            )

        if self.args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load and don't log every step
            if self.total_steps_done % self.config.tb_plot_step == 0:
                iter_stats = log_dict
                iter_stats.update(loss_dict)
                self.tb_logger.tb_train_step_stats(self.total_steps_done, iter_stats)
            if self.total_steps_done % self.config.save_step == 0 and self.total_steps_done != 0:
                if self.config.checkpoint:
                    # checkpoint the model
                    model_loss = (
                        loss_dict[self.config.target_loss] if "target_loss" in self.config else loss_dict["loss"]
                    )
                    save_checkpoint(
                        self.config,
                        self.model,
                        self.optimizer,
                        self.scaler if self.use_amp_scaler else None,
                        self.total_steps_done,
                        self.epochs_done,
                        self.output_path,
                        model_loss=model_loss,
                    )
                # training visualizations
                figures, audios = None, None
                if hasattr(self.model, "module") and hasattr(self.model.module, "train_log"):
                    figures, audios = self.model.module.train_log(self.ap, batch, outputs)
                elif hasattr(self.model, "train_log"):
                    figures, audios = self.model.train_log(self.ap, batch, outputs)
                if figures is not None:
                    self.tb_logger.tb_train_figures(self.total_steps_done, figures)
                if audios is not None:
                    self.tb_logger.tb_train_audios(self.total_steps_done, audios, self.ap.sample_rate)
        self.total_steps_done += 1
        self.callbacks.on_train_step_end()
        return outputs, loss_dict

    def train_epoch(self) -> None:
        """Main entry point for the training loop. Run training on the all training samples."""
        self.train_loader = self.get_train_dataloader(
            self.ap,
            self.data_train,
            verbose=True,
        )
        self.model.train()
        epoch_start_time = time.time()
        if self.use_cuda:
            batch_num_steps = int(len(self.train_loader.dataset) / (self.config.batch_size * self.num_gpus))
        else:
            batch_num_steps = int(len(self.train_loader.dataset) / self.config.batch_size)
        self.c_logger.print_train_start()
        for cur_step, batch in enumerate(self.train_loader):
            loader_start_time = time.time()
            _, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
        epoch_time = time.time() - epoch_start_time
        # Plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.tb_logger.tb_train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.tb_model_param_stats:
                self.tb_logger.tb_model_weights(self.model, self.total_steps_done)

    @staticmethod
    def _model_eval_step(
        batch: Dict, model: nn.Module, criterion: nn.Module, optimizer_idx: int = None
    ) -> Tuple[Dict, Dict]:
        """
        Perform a evaluation forward pass. Compute model outputs and losses with no gradients.

        Args:
            batch (Dict): IBatch of inputs.
            model (nn.Module): Model to call evaluation.
            criterion (nn.Module): Model criterion.
            optimizer_idx (int, optional): Optimizer ID to define the closure in multi-optimizer training. Defaults to None.

        Returns:
            Tuple[Dict, Dict]: model outputs and losses.
        """
        input_args = [batch, criterion]
        if optimizer_idx is not None:
            input_args.append(optimizer_idx)
        if hasattr(model, "module"):
            return model.module.eval_step(*input_args)
        return model.eval_step(*input_args)

    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        """Perform a evaluation step on a batch of inputs and log the process.

        Args:
            batch (Dict): Input batch.
            step (int): Current step number in this epoch.

        Returns:
            Tuple[Dict, Dict]: Model outputs and losses.
        """
        with torch.no_grad():
            outputs_per_optimizer = None
            loss_dict = {}
            if not isinstance(self.optimizer, list):
                outputs, loss_dict = self._model_eval_step(batch, self.model, self.criterion)
            else:
                outputs_per_optimizer = [None] * len(self.optimizer)
                for idx, _ in enumerate(self.optimizer):
                    criterion = self.criterion
                    outputs, loss_dict_new = self._model_eval_step(batch, self.model, criterion, idx)
                    outputs_per_optimizer[idx] = outputs
                    if loss_dict_new is not None:
                        loss_dict.update(loss_dict_new)
                outputs = outputs_per_optimizer

            # update avg stats
            update_eval_values = dict()
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            self.keep_avg_eval.update_values(update_eval_values)

            if self.config.print_eval:
                self.c_logger.print_eval_step(step, loss_dict, self.keep_avg_eval.avg_values)
        return outputs, loss_dict

    def eval_epoch(self) -> None:
        """Main entry point for the evaluation loop. Run evaluation on the all validation samples."""
        self.eval_loader = (
            self.get_eval_dataloader(
                self.ap,
                self.data_eval,
                verbose=True,
            )
            if self.config.run_eval
            else None
        )

        self.model.eval()
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        batch = None
        for cur_step, batch in enumerate(self.eval_loader):
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({"avg_loader_time": loader_time})
            outputs, _ = self.eval_step(batch, cur_step)
        # plot epoch stats, artifacts and figures
        if self.args.rank == 0:
            figures, audios = None, None
            if hasattr(self.model, "module") and hasattr(self.model.module, "eval_log"):
                figures, audios = self.model.module.eval_log(self.ap, batch, outputs)
            elif hasattr(self.model, "eval_log"):
                figures, audios = self.model.eval_log(self.ap, batch, outputs)
            if figures is not None:
                self.tb_logger.tb_eval_figures(self.total_steps_done, figures)
            if audios is not None:
                self.tb_logger.tb_eval_audios(self.total_steps_done, audios, self.ap.sample_rate)
            self.tb_logger.tb_eval_stats(self.total_steps_done, self.keep_avg_eval.avg_values)

    def test_run(self) -> None:
        """Run test and log the results. Test run must be defined by the model.
        Model must return figures and audios to be logged by the Tensorboard."""
        if hasattr(self.model, "test_run"):
            if hasattr(self.eval_loader.load_test_samples):
                samples = self.eval_loader.load_test_samples(1)
                figures, audios = self.model.test_run(samples)
            else:
                figures, audios = self.model.test_run()
            self.tb_logger.tb_test_audios(self.total_steps_done, audios, self.config.audio["sample_rate"])
            self.tb_logger.tb_test_figures(self.total_steps_done, figures)

    def _fit(self) -> None:
        """🏃 train -> evaluate -> test for the number of epochs."""
        if self.restore_step != 0 or self.args.best_path:
            print(" > Restoring best loss from " f"{os.path.basename(self.args.best_path)} ...")
            self.best_loss = torch.load(self.args.best_path, map_location="cpu")["model_loss"]
            print(f" > Starting with loaded last best loss {self.best_loss}.")

        self.total_steps_done = self.restore_step

        for epoch in range(0, self.config.epochs):
            self.callbacks.on_epoch_start()
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs)
            self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs and self.args.rank < 0:
                self.test_run()
            self.c_logger.print_epoch_end(
                epoch, self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values
            )
            self.save_best_model()
            self.callbacks.on_epoch_end()

    def fit(self) -> None:
        """Where the ✨️magic✨️ happens..."""
        try:
            self._fit()
        except KeyboardInterrupt:
            self.callbacks.on_keyboard_interrupt()
            # if the output folder is empty remove the run.
            remove_experiment_folder(self.output_path)
            # stop without error signal
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)  # pylint: disable=protected-access
        except BaseException:  # pylint: disable=broad-except
            remove_experiment_folder(self.output_path)
            traceback.print_exc()
            sys.exit(1)

    def save_best_model(self) -> None:
        """Save the best model. It only saves if the current target loss is smaller then the previous."""
        self.best_loss = save_best_model(
            self.keep_avg_eval["avg_loss"] if self.keep_avg_eval else self.keep_avg_train["avg_loss"],
            self.best_loss,
            self.config,
            self.model,
            self.optimizer,
            self.scaler if self.use_amp_scaler else None,
            self.total_steps_done,
            self.epochs_done,
            self.output_path,
            keep_all_best=self.config.keep_all_best,
            keep_after=self.config.keep_after,
        )

    @staticmethod
    def _setup_logger_config(log_file: str) -> None:
        logging.basicConfig(
            level=logging.INFO, format="", handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

    @staticmethod
    def _is_apex_available() -> bool:
        """Check if Nvidia's APEX is available."""
        return importlib.util.find_spec("apex") is not None

    @staticmethod
    def get_optimizer(model: nn.Module, config: Coqpit) -> Union[torch.optim.Optimizer, List]:
        """Receive the optimizer from the model if model implements `get_optimizer()` else
        check the optimizer parameters in the config and try initiating the optimizer.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A optimizer or a list of optimizers. GAN models define a list.
        """
        if hasattr(model, "get_optimizer"):
            optimizer = model.get_optimizer()
        if optimizer is None:
            optimizer_name = config.optimizer
            optimizer_params = config.optimizer_params
            return get_optimizer(optimizer_name, optimizer_params, config.lr, model)
        return optimizer

    @staticmethod
    def get_lr(model: nn.Module, config: Coqpit) -> Union[float, List[float]]:
        """Set the initial learning rate by the model if model implements `get_lr()` else try setting the learning rate
        fromthe config.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[float, List[float]]: A single learning rate or a list of learning rates, one for each optimzier.
        """
        lr = None
        if hasattr(model, "get_lr"):
            lr = model.get_lr()
        if lr is None:
            lr = config.lr
        return lr

    @staticmethod
    def get_scheduler(
        model: nn.Module, config: Coqpit, optimizer: Union[torch.optim.Optimizer, List]
    ) -> Union[torch.optim.lr_scheduler._LRScheduler, List]:  # pylint: disable=protected-access
        """Receive the scheduler from the model if model implements `get_scheduler()` else
        check the config and try initiating the scheduler.

        Args:
            model (nn.Module): Training model.
            config (Coqpit): Training configuration.

        Returns:
            Union[torch.optim.Optimizer, List]: A scheduler or a list of schedulers, one for each optimizer.
        """
        scheduler = None
        if hasattr(model, "get_scheduler"):
            scheduler = model.get_scheduler(optimizer)
        if scheduler is None:
            lr_scheduler = config.lr_scheduler
            lr_scheduler_params = config.lr_scheduler_params
            return get_scheduler(lr_scheduler, lr_scheduler_params, optimizer)
        return scheduler

    @staticmethod
    def get_criterion(model: nn.Module) -> nn.Module:
        """Receive the criterion from the model. Model must implement `get_criterion()`.

        Args:
            model (nn.Module): Training model.

        Returns:
            nn.Module: Criterion layer.
        """
        criterion = None
        criterion = model.get_criterion()
        return criterion


def init_arguments():
    train_config = TrainingArgs()
    parser = train_config.init_argparse(arg_prefix="")
    return parser


def get_last_checkpoint(path):
    """Get latest checkpoint or/and best model in path.

    It is based on globbing for `*.pth.tar` and the RegEx
    `(checkpoint|best_model)_([0-9]+)`.

    Args:
        path (list): Path to files to be compared.

    Raises:
        ValueError: If no checkpoint or best_model files are found.

    Returns:
        last_checkpoint (str): Last checkpoint filename.
    """
    file_names = glob.glob(os.path.join(path, "*.pth.tar"))
    last_models = {}
    last_model_nums = {}
    for key in ["checkpoint", "best_model"]:
        last_model_num = None
        last_model = None
        # pass all the checkpoint files and find
        # the one with the largest model number suffix.
        for file_name in file_names:
            match = re.search(f"{key}_([0-9]+)", file_name)
            if match is not None:
                model_num = int(match.groups()[0])
                if last_model_num is None or model_num > last_model_num:
                    last_model_num = model_num
                    last_model = file_name

        # if there is not checkpoint found above
        # find the checkpoint with the latest
        # modification date.
        key_file_names = [fn for fn in file_names if key in fn]
        if last_model is None and len(key_file_names) > 0:
            last_model = max(key_file_names, key=os.path.getctime)
            last_model_num = torch.load(last_model)["step"]

        if last_model is not None:
            last_models[key] = last_model
            last_model_nums[key] = last_model_num

    # check what models were found
    if not last_models:
        raise ValueError(f"No models found in continue path {path}!")
    if "checkpoint" not in last_models:  # no checkpoint just best model
        last_models["checkpoint"] = last_models["best_model"]
    elif "best_model" not in last_models:  # no best model
        # this shouldn't happen, but let's handle it just in case
        last_models["best_model"] = None
    # finally check if last best model is more recent than checkpoint
    elif last_model_nums["best_model"] > last_model_nums["checkpoint"]:
        last_models["checkpoint"] = last_models["best_model"]

    return last_models["checkpoint"], last_models["best_model"]


def process_args(args, config=None):
    """Process parsed comand line arguments and initialize the config if not provided.

    Args:
        args (argparse.Namespace or dict like): Parsed input arguments.
        config (Coqpit): Model config. If none, it is generated from `args`. Defaults to None.

    Returns:
        c (TTS.utils.io.AttrDict): Config paramaters.
        out_path (str): Path to save models and logging.
        audio_path (str): Path to save generated test audios.
        c_logger (TTS.utils.console_logger.ConsoleLogger): Class that does
            logging to the console.
        tb_logger (TTS.utils.tensorboard.TensorboardLogger): Class that does
            the TensorBoard logging.

    TODO:
        - Interactive config definition.
    """
    if isinstance(args, tuple):
        args, coqpit_overrides = args
    if args.continue_path:
        # continue a previous training from its output folder
        experiment_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, "config.json")
        args.restore_path, best_model = get_last_checkpoint(args.continue_path)
        if not args.best_path:
            args.best_path = best_model
    # init config if not already defined
    if config is None:
        if args.config_path:
            # init from a file
            config = load_config(args.config_path)
        else:
            # init from console args
            from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

            config_base = BaseTrainingConfig()
            config_base.parse_known_args(coqpit_overrides)
            config = register_config(config_base.model)()
    # override values from command-line args
    config.parse_known_args(coqpit_overrides, relaxed_parser=True)
    if config.mixed_precision:
        print("   >  Mixed precision mode is ON")
    experiment_path = args.continue_path
    if not experiment_path:
        experiment_path = create_experiment_folder(config.output_path, config.run_name)
    audio_path = os.path.join(experiment_path, "test_audios")
    # setup rank 0 process in distributed training
    tb_logger = None
    if args.rank == 0:
        os.makedirs(audio_path, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        # if model characters are not set in the config file
        # save the default set to the config file for future
        # compatibility.
        if config.has("characters_config"):
            used_characters = parse_symbols()
            new_fields["characters"] = used_characters
        copy_model_files(config, experiment_path, new_fields)
        os.chmod(audio_path, 0o775)
        os.chmod(experiment_path, 0o775)
        tb_logger = TensorboardLogger(experiment_path, model_name=config.model)
        # write model desc to tensorboard
        tb_logger.tb_add_text("model-config", f"<pre>{config.to_json()}</pre>", 0)
    c_logger = ConsoleLogger()
    return config, experiment_path, audio_path, c_logger, tb_logger


def init_training(argv: Union[List, Coqpit], config: Coqpit = None):
    """Initialization of a training run."""
    if isinstance(argv, Coqpit):
        parser = argv.init_argparse(arg_prefix="")
    else:
        parser = init_arguments()
    args = parser.parse_known_args()
    config, OUT_PATH, AUDIO_PATH, c_logger, tb_logger = process_args(args, config)
    return args[0], config, OUT_PATH, AUDIO_PATH, c_logger, tb_logger
