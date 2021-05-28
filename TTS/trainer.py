# -*- coding: utf-8 -*-

import importlib
import logging
import os
import time
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Union

import torch
from coqpit import Coqpit

# DISTRIBUTED
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.tts.datasets import TTSDataset, load_meta_data
from TTS.tts.layers import setup_loss
from TTS.tts.models import setup_model
from TTS.tts.utils.io import save_best_model, save_checkpoint
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.distribute import init_distributed
from TTS.utils.generic_utils import KeepAverage, count_parameters, set_init_dict
from TTS.utils.logging import ConsoleLogger, TensorboardLogger
from TTS.utils.training import check_update, setup_torch_training_env


@dataclass
class TrainingArgs(Coqpit):
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


# pylint: disable=import-outside-toplevel, too-many-public-methods
class TrainerTTS:
    use_cuda, num_gpus = setup_torch_training_env(True, False)

    def __init__(
        self,
        args: Union[Coqpit, Namespace],
        config: Coqpit,
        c_logger: ConsoleLogger,
        tb_logger: TensorboardLogger,
        model: nn.Module = None,
        output_path: str = None,
    ) -> None:
        self.args = args
        self.config = config
        self.c_logger = c_logger
        self.tb_logger = tb_logger
        self.output_path = output_path

        self.total_steps_done = 0
        self.epochs_done = 0
        self.restore_step = 0
        self.best_loss = float("inf")
        self.train_loader = None
        self.eval_loader = None
        self.output_audio_path = os.path.join(output_path, "test_audios")

        self.keep_avg_train = None
        self.keep_avg_eval = None

        log_file = os.path.join(self.output_path, f"trainer_{args.rank}_log.txt")
        self._setup_logger_config(log_file)

        # model, audio processor, datasets, loss
        # init audio processor
        self.ap = AudioProcessor(**self.config.audio.to_dict())

        # init character processor
        self.model_characters = self.get_character_processor(self.config)

        # load dataset samples
        self.data_train, self.data_eval = load_meta_data(self.config.datasets)

        # default speaker manager
        self.speaker_manager = self.get_speaker_manager(
            self.config, args.restore_path, self.config.output_path, self.data_train
        )

        # init TTS model
        if model is not None:
            self.model = model
        else:
            self.model = self.get_model(
                len(self.model_characters),
                self.speaker_manager.num_speakers,
                self.config,
                self.speaker_manager.x_vector_dim if self.speaker_manager.x_vectors else None,
            )

        # setup criterion
        self.criterion = self.get_criterion(self.config)

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        # DISTRUBUTED
        if self.num_gpus > 1:
            init_distributed(
                args.rank,
                self.num_gpus,
                args.group_id,
                self.config.distributed["backend"],
                self.config.distributed["url"],
            )

        # scalers for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler() if self.config.mixed_precision and self.use_cuda else None

        # setup optimizer
        self.optimizer = self.get_optimizer(self.model, self.config)

        if self.args.restore_path:
            self.model, self.optimizer, self.scaler, self.restore_step = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer, self.scaler
            )

        # setup scheduler
        self.scheduler = self.get_scheduler(self.config, self.optimizer)

        # DISTRUBUTED
        if self.num_gpus > 1:
            self.model = DDP_th(self.model, device_ids=[args.rank])

        # count model size
        num_params = count_parameters(self.model)
        print("\n > Model has {} parameters".format(num_params))

    @staticmethod
    def get_model(num_chars: int, num_speakers: int, config: Coqpit, x_vector_dim: int) -> nn.Module:
        model = setup_model(num_chars, num_speakers, config, x_vector_dim)
        return model

    @staticmethod
    def get_optimizer(model: nn.Module, config: Coqpit) -> torch.optim.Optimizer:
        optimizer_name = config.optimizer
        optimizer_params = config.optimizer_params
        if optimizer_name.lower() == "radam":
            module = importlib.import_module("TTS.utils.radam")
            optimizer = getattr(module, "RAdam")
        else:
            optimizer = getattr(torch.optim, optimizer_name)
        return optimizer(model.parameters(), lr=config.lr, **optimizer_params)

    @staticmethod
    def get_character_processor(config: Coqpit) -> str:
        # setup custom characters if set in config file.
        # TODO: implement CharacterProcessor
        if config.characters is not None:
            symbols, phonemes = make_symbols(**config.characters.to_dict())
        else:
            from TTS.tts.utils.text.symbols import phonemes, symbols
        model_characters = phonemes if config.use_phonemes else symbols
        return model_characters

    @staticmethod
    def get_speaker_manager(
        config: Coqpit, restore_path: str = "", out_path: str = "", data_train: List = None
    ) -> SpeakerManager:
        speaker_manager = SpeakerManager()
        if config.use_speaker_embedding:
            if restore_path:
                speakers_file = os.path.join(os.path.dirname(restore_path), "speaker.json")
                if not os.path.exists(speakers_file):
                    print(
                        "WARNING: speakers.json was not found in restore_path, trying to use CONFIG.external_speaker_embedding_file"
                    )
                    speakers_file = config.external_speaker_embedding_file

                if config.use_external_speaker_embedding_file:
                    speaker_manager.load_x_vectors_file(speakers_file)
                else:
                    speaker_manager.load_ids_file(speakers_file)
            elif config.use_external_speaker_embedding_file and config.external_speaker_embedding_file:
                speaker_manager.load_x_vectors_file(config.external_speaker_embedding_file)
            else:
                speaker_manager.parse_speakers_from_items(data_train)
                file_path = os.path.join(out_path, "speakers.json")
                speaker_manager.save_ids_file(file_path)
        return speaker_manager

    @staticmethod
    def get_scheduler(
        config: Coqpit, optimizer: torch.optim.Optimizer
    ) -> torch.optim.lr_scheduler._LRScheduler:  # pylint: disable=protected-access
        lr_scheduler = config.lr_scheduler
        lr_scheduler_params = config.lr_scheduler_params
        if lr_scheduler is None:
            return None
        if lr_scheduler.lower() == "noamlr":
            from TTS.utils.training import NoamLR

            scheduler = NoamLR
        else:
            scheduler = getattr(torch.optim, lr_scheduler)
        return scheduler(optimizer, **lr_scheduler_params)

    @staticmethod
    def get_criterion(config: Coqpit) -> nn.Module:
        return setup_loss(config)

    def restore_model(
        self,
        config: Coqpit,
        restore_path: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scaler: torch.cuda.amp.GradScaler = None,
    ) -> Tuple[nn.Module, torch.optim.Optimizer, torch.cuda.amp.GradScaler, int]:
        print(" > Restoring from %s ..." % os.path.basename(restore_path))
        checkpoint = torch.load(restore_path)
        try:
            print(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            print(" > Restoring Optimizer...")
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint and config.mixed_precision:
                print(" > Restoring AMP Scaler...")
                scaler.load_state_dict(checkpoint["scaler"])
        except (KeyError, RuntimeError):
            print(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        for group in optimizer.param_groups:
            group["lr"] = self.config.lr
        print(
            " > Model restored from step %d" % checkpoint["step"],
        )
        restore_step = checkpoint["step"]
        return model, optimizer, scaler, restore_step

    def _get_loader(
        self,
        r: int,
        ap: AudioProcessor,
        is_eval: bool,
        data_items: List,
        verbose: bool,
        speaker_mapping: Union[Dict, List],
    ) -> DataLoader:
        if is_eval and not self.config.run_eval:
            loader = None
        else:
            dataset = TTSDataset(
                outputs_per_step=r,
                text_cleaner=self.config.text_cleaner,
                compute_linear_spec=self.config.model.lower() == "tacotron",
                meta_data=data_items,
                ap=ap,
                tp=self.config.characters,
                add_blank=self.config["add_blank"],
                batch_group_size=0 if is_eval else self.config.batch_group_size * self.config.batch_size,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                phoneme_cache_path=self.config.phoneme_cache_path,
                use_phonemes=self.config.use_phonemes,
                phoneme_language=self.config.phoneme_language,
                enable_eos_bos=self.config.enable_eos_bos_chars,
                use_noise_augment=not is_eval,
                verbose=verbose,
                speaker_mapping=speaker_mapping
                if self.config.use_speaker_embedding and self.config.use_external_speaker_embedding_file
                else None,
            )

            if self.config.use_phonemes and self.config.compute_input_seq_cache:
                # precompute phonemes to have a better estimate of sequence lengths.
                dataset.compute_input_seq(self.config.num_loader_workers)
            dataset.sort_items()

            sampler = DistributedSampler(dataset) if self.num_gpus > 1 else None
            loader = DataLoader(
                dataset,
                batch_size=self.config.eval_batch_size if is_eval else self.config.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                drop_last=False,
                sampler=sampler,
                num_workers=self.config.num_val_loader_workers if is_eval else self.config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def get_train_dataloader(
        self, r: int, ap: AudioProcessor, data_items: List, verbose: bool, speaker_mapping: Union[List, Dict]
    ) -> DataLoader:
        return self._get_loader(r, ap, False, data_items, verbose, speaker_mapping)

    def get_eval_dataloder(
        self, r: int, ap: AudioProcessor, data_items: List, verbose: bool, speaker_mapping: Union[List, Dict]
    ) -> DataLoader:
        return self._get_loader(r, ap, True, data_items, verbose, speaker_mapping)

    def format_batch(self, batch: List) -> Dict:
        # setup input batch
        text_input = batch[0]
        text_lengths = batch[1]
        speaker_names = batch[2]
        linear_input = batch[3] if self.config.model.lower() in ["tacotron"] else None
        mel_input = batch[4]
        mel_lengths = batch[5]
        stop_targets = batch[6]
        item_idx = batch[7]
        speaker_embeddings = batch[8]
        attn_mask = batch[9]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())

        # convert speaker names to ids
        if self.config.use_speaker_embedding:
            if self.config.use_external_speaker_embedding_file:
                speaker_embeddings = batch[8]
                speaker_ids = None
            else:
                speaker_ids = [self.speaker_manager.speaker_ids[speaker_name] for speaker_name in speaker_names]
                speaker_ids = torch.LongTensor(speaker_ids)
                speaker_embeddings = None
        else:
            speaker_embeddings = None
            speaker_ids = None

        # compute durations from attention masks
        if attn_mask is not None:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, : text_lengths[idx], : mel_lengths[idx]].max(1)[1]
                # c_idxs, counts = torch.unique_consecutive(c_idxs, return_counts=True)
                c_idxs, counts = torch.unique(c_idxs, return_counts=True)
                dur = torch.ones([text_lengths[idx]]).to(counts.dtype)
                dur[c_idxs] = counts
                # smooth the durations and set any 0 duration to 1
                # by cutting off from the largest duration indeces.
                extra_frames = dur.sum() - mel_lengths[idx]
                largest_idxs = torch.argsort(-dur)[:extra_frames]
                dur[largest_idxs] -= 1
                assert (
                    dur.sum() == mel_lengths[idx]
                ), f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
                durations[idx, : text_lengths[idx]] = dur

        # set stop targets view, we predict a single stop token per iteration.
        stop_targets = stop_targets.view(text_input.shape[0], stop_targets.size(1) // self.config.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float().squeeze(2)

        # dispatch batch to GPU
        if self.use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            linear_input = linear_input.cuda(non_blocking=True) if self.config.model.lower() in ["tacotron"] else None
            stop_targets = stop_targets.cuda(non_blocking=True)
            attn_mask = attn_mask.cuda(non_blocking=True) if attn_mask is not None else None
            durations = durations.cuda(non_blocking=True) if attn_mask is not None else None
            if speaker_ids is not None:
                speaker_ids = speaker_ids.cuda(non_blocking=True)
            if speaker_embeddings is not None:
                speaker_embeddings = speaker_embeddings.cuda(non_blocking=True)

        return {
            "text_input": text_input,
            "text_lengths": text_lengths,
            "mel_input": mel_input,
            "mel_lengths": mel_lengths,
            "linear_input": linear_input,
            "stop_targets": stop_targets,
            "attn_mask": attn_mask,
            "durations": durations,
            "speaker_ids": speaker_ids,
            "x_vectors": speaker_embeddings,
            "max_text_length": max_text_length,
            "max_spec_length": max_spec_length,
            "item_idx": item_idx,
        }

    def train_step(self, batch: Dict, batch_n_steps: int, step: int, loader_start_time: float) -> Tuple[Dict, Dict]:
        self.on_train_step_start()
        step_start_time = time.time()

        # format data
        batch = self.format_batch(batch)
        loader_time = time.time() - loader_start_time

        # zero-out optimizer
        self.optimizer.zero_grad()

        with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
            outputs, loss_dict = self.model.train_step(batch, self.criterion)

        # check nan loss
        if torch.isnan(loss_dict["loss"]).any():
            raise RuntimeError(f"Detected NaN loss at step {self.total_steps_done}.")

        # optimizer step
        if self.config.mixed_precision:
            # model optimizer step in mixed precision mode
            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm, _ = check_update(self.model, self.config.grad_clip, ignore_stopnet=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # main model optimizer step
            loss_dict["loss"].backward()
            grad_norm, _ = check_update(self.model, self.config.grad_clip, ignore_stopnet=True)
            self.optimizer.step()

        step_time = time.time() - step_start_time

        # setup lr
        if self.config.lr_scheduler:
            self.scheduler.step()

        # detach loss values
        loss_dict_new = dict()
        for key, value in loss_dict.items():
            if isinstance(value, (int, float)):
                loss_dict_new[key] = value
            else:
                loss_dict_new[key] = value.item()
        loss_dict = loss_dict_new

        # update avg stats
        update_train_values = dict()
        for key, value in loss_dict.items():
            update_train_values["avg_" + key] = value
        update_train_values["avg_loader_time"] = loader_time
        update_train_values["avg_step_time"] = step_time
        self.keep_avg_train.update_values(update_train_values)

        # print training progress
        current_lr = self.optimizer.param_groups[0]["lr"]
        if self.total_steps_done % self.config.print_step == 0:
            log_dict = {
                "max_spec_length": [batch["max_spec_length"], 1],  # value, precision
                "max_text_length": [batch["max_text_length"], 1],
                "step_time": [step_time, 4],
                "loader_time": [loader_time, 2],
                "current_lr": current_lr,
            }
            self.c_logger.print_train_step(
                batch_n_steps, step, self.total_steps_done, log_dict, loss_dict, self.keep_avg_train.avg_values
            )

        if self.args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load
            if self.total_steps_done % self.config.tb_plot_step == 0:
                iter_stats = {
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "step_time": step_time,
                }
                iter_stats.update(loss_dict)
                self.tb_logger.tb_train_step_stats(self.total_steps_done, iter_stats)

            if self.total_steps_done % self.config.save_step == 0:
                if self.config.checkpoint:
                    # save model
                    save_checkpoint(
                        self.model,
                        self.optimizer,
                        self.total_steps_done,
                        self.epochs_done,
                        self.config.r,
                        self.output_path,
                        model_loss=loss_dict["loss"],
                        characters=self.model_characters,
                        scaler=self.scaler.state_dict() if self.config.mixed_precision else None,
                    )
                # training visualizations
                figures, audios = self.model.train_log(self.ap, batch, outputs)
                self.tb_logger.tb_train_figures(self.total_steps_done, figures)
                self.tb_logger.tb_train_audios(self.total_steps_done, {"TrainAudio": audios}, self.ap.sample_rate)
        self.total_steps_done += 1
        self.on_train_step_end()
        return outputs, loss_dict

    def train_epoch(self) -> None:
        self.model.train()
        epoch_start_time = time.time()
        if self.use_cuda:
            batch_num_steps = int(len(self.train_loader.dataset) / (self.config.batch_size * self.num_gpus))
        else:
            batch_num_steps = int(len(self.train_loader.dataset) / self.config.batch_size)
        self.c_logger.print_train_start()
        loader_start_time = time.time()
        for cur_step, batch in enumerate(self.train_loader):
            _, _ = self.train_step(batch, batch_num_steps, cur_step, loader_start_time)
        epoch_time = time.time() - epoch_start_time
        # Plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.tb_logger.tb_train_epoch_stats(self.total_steps_done, epoch_stats)
            if self.config.tb_model_param_stats:
                self.tb_logger.tb_model_weights(self.model, self.total_steps_done)

    def eval_step(self, batch: Dict, step: int) -> Tuple[Dict, Dict]:
        with torch.no_grad():
            step_start_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs, loss_dict = self.model.eval_step(batch, self.criterion)

            step_time = time.time() - step_start_time

            # detach loss values
            loss_dict_new = dict()
            for key, value in loss_dict.items():
                if isinstance(value, (int, float)):
                    loss_dict_new[key] = value
                else:
                    loss_dict_new[key] = value.item()
            loss_dict = loss_dict_new

            # update avg stats
            update_eval_values = dict()
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            update_eval_values["avg_step_time"] = step_time
            self.keep_avg_eval.update_values(update_eval_values)

            if self.config.print_eval:
                self.c_logger.print_eval_step(step, loss_dict, self.keep_avg_eval.avg_values)
        return outputs, loss_dict

    def eval_epoch(self) -> None:
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
        # Plot epoch stats and samples from the last batch.
        if self.args.rank == 0:
            figures, eval_audios = self.model.eval_log(self.ap, batch, outputs)
            self.tb_logger.tb_eval_figures(self.total_steps_done, figures)
            self.tb_logger.tb_eval_audios(self.total_steps_done, {"EvalAudio": eval_audios}, self.ap.sample_rate)

    def test_run(
        self,
    ) -> None:
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        cond_inputs = self._get_cond_inputs()
        for idx, sen in enumerate(test_sentences):
            wav, alignment, model_outputs, _ = synthesis(
                self.model,
                sen,
                self.config,
                self.use_cuda,
                self.ap,
                speaker_id=cond_inputs["speaker_id"],
                x_vector=cond_inputs["x_vector"],
                style_wav=cond_inputs["style_wav"],
                enable_eos_bos_chars=self.config.enable_eos_bos_chars,
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()

            file_path = os.path.join(self.output_audio_path, str(self.total_steps_done))
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, "TestSentence_{}.wav".format(idx))
            self.ap.save_wav(wav, file_path)
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(model_outputs, self.ap, output_fig=False)
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment, output_fig=False)

        self.tb_logger.tb_test_audios(self.total_steps_done, test_audios, self.config.audio["sample_rate"])
        self.tb_logger.tb_test_figures(self.total_steps_done, test_figures)

    def _get_cond_inputs(self) -> Dict:
        # setup speaker_id
        speaker_id = 0 if self.config.use_speaker_embedding else None
        # setup x_vector
        x_vector = (
            self.speaker_manager.get_x_vectors_by_speaker(self.speaker_manager.speaker_ids[0])
            if self.config.use_external_speaker_embedding_file and self.config.use_speaker_embedding
            else None
        )
        # setup style_mel
        if self.config.has("gst_style_input"):
            style_wav = self.config.gst_style_input
        else:
            style_wav = None
        if style_wav is None and "use_gst" in self.config and self.config.use_gst:
            # inicialize GST with zero dict.
            style_wav = {}
            print("WARNING: You don't provided a gst style wav, for this reason we use a zero tensor!")
            for i in range(self.config.gst["gst_num_style_tokens"]):
                style_wav[str(i)] = 0
        cond_inputs = {"speaker_id": speaker_id, "style_wav": style_wav, "x_vector": x_vector}
        return cond_inputs

    def fit(self) -> None:
        if self.restore_step != 0 or self.args.best_path:
            print(" > Restoring best loss from " f"{os.path.basename(self.args.best_path)} ...")
            self.best_loss = torch.load(self.args.best_path, map_location="cpu")["model_loss"]
            print(f" > Starting with loaded last best loss {self.best_loss}.")

        # define data loaders
        self.train_loader = self.get_train_dataloader(
            self.config.r, self.ap, self.data_train, verbose=True, speaker_mapping=self.speaker_manager.speaker_ids
        )
        self.eval_loader = (
            self.get_eval_dataloder(
                self.config.r, self.ap, self.data_train, verbose=True, speaker_mapping=self.speaker_manager.speaker_ids
            )
            if self.config.run_eval
            else None
        )

        self.total_steps_done = self.restore_step

        for epoch in range(0, self.config.epochs):
            self.on_epoch_start()
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage() if self.config.run_eval else None
            self.epochs_done = epoch
            self.c_logger.print_epoch_start(epoch, self.config.epochs)
            self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            if epoch >= self.config.test_delay_epochs:
                self.test_run()
            self.c_logger.print_epoch_end(
                epoch, self.keep_avg_eval.avg_values if self.config.run_eval else self.keep_avg_train.avg_values
            )
            self.save_best_model()
            self.on_epoch_end()

    def save_best_model(self) -> None:
        self.best_loss = save_best_model(
            self.keep_avg_eval["avg_loss"] if self.keep_avg_eval else self.keep_avg_train["avg_loss"],
            self.best_loss,
            self.model,
            self.optimizer,
            self.total_steps_done,
            self.epochs_done,
            self.config.r,
            self.output_path,
            self.model_characters,
            keep_all_best=self.config.keep_all_best,
            keep_after=self.config.keep_after,
            scaler=self.scaler.state_dict() if self.config.mixed_precision else None,
        )

    @staticmethod
    def _setup_logger_config(log_file: str) -> None:
        logging.basicConfig(
            level=logging.INFO, format="", handlers=[logging.FileHandler(log_file), logging.StreamHandler()]
        )

    def on_epoch_start(self) -> None:  # pylint: disable=no-self-use
        if hasattr(self.model, "on_epoch_start"):
            self.model.on_epoch_start(self)

        if hasattr(self.criterion, "on_epoch_start"):
            self.criterion.on_epoch_start(self)

        if hasattr(self.optimizer, "on_epoch_start"):
            self.optimizer.on_epoch_start(self)

    def on_epoch_end(self) -> None:  # pylint: disable=no-self-use
        if hasattr(self.model, "on_epoch_end"):
            self.model.on_epoch_end(self)

        if hasattr(self.criterion, "on_epoch_end"):
            self.criterion.on_epoch_end(self)

        if hasattr(self.optimizer, "on_epoch_end"):
            self.optimizer.on_epoch_end(self)

    def on_train_step_start(self) -> None:  # pylint: disable=no-self-use
        if hasattr(self.model, "on_train_step_start"):
            self.model.on_train_step_start(self)

        if hasattr(self.criterion, "on_train_step_start"):
            self.criterion.on_train_step_start(self)

        if hasattr(self.optimizer, "on_train_step_start"):
            self.optimizer.on_train_step_start(self)

    def on_train_step_end(self) -> None:  # pylint: disable=no-self-use
        if hasattr(self.model, "on_train_step_end"):
            self.model.on_train_step_end(self)

        if hasattr(self.criterion, "on_train_step_end"):
            self.criterion.on_train_step_end(self)

        if hasattr(self.optimizer, "on_train_step_end"):
            self.optimizer.on_train_step_end(self)
