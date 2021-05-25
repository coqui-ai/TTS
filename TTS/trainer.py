# -*- coding: utf-8 -*-

import os
import sys
import time
import traceback
from random import randrange
import logging
import importlib

import numpy as np
import torch

# DISTRIBUTED
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from TTS.tts.datasets import load_meta_data, TTSDataset
from TTS.tts.layers import setup_loss
from TTS.tts.models import setup_model
from TTS.tts.utils.io import save_best_model, save_checkpoint
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.arguments import init_training
from TTS.tts.utils.visual import plot_spectrogram, plot_alignment
from TTS.utils.audio import AudioProcessor
from TTS.utils.distribute import init_distributed, reduce_tensor
from TTS.utils.generic_utils import KeepAverage, count_parameters, remove_experiment_folder, set_init_dict, find_module
from TTS.utils.training import setup_torch_training_env, check_update


class TrainerTTS:
    use_cuda, num_gpus = setup_torch_training_env(True, False)

    def __init__(self,
                 args,
                 config,
                 c_logger,
                 tb_logger,
                 model=None,
                 output_path=None):
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
        self.output_audio_path = os.path.join(output_path, 'test_audios')

        self.keep_avg_train = None
        self.keep_avg_eval = None

        # model, audio processor, datasets, loss
        # init audio processor
        self.ap = AudioProcessor(**config.audio.to_dict())

        # init character processor
        self.model_characters = self.init_character_processor()

        # load dataset samples
        self.data_train, self.data_eval = load_meta_data(config.datasets)

        # default speaker manager
        self.speaker_manager = self.init_speaker_manager()

        # init TTS model
        if model is not None:
            self.model = model
        else:
            self.model = self.init_model()

        # setup criterion
        self.criterion = self.init_criterion()

        # DISTRUBUTED
        if self.num_gpus > 1:
            init_distributed(args.rank, self.num_gpus, args.group_id,
                             config.distributed["backend"],
                             config.distributed["url"])

        # scalers for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(
        ) if config.mixed_precision else None

        # setup optimizer
        self.optimizer = self.init_optimizer(self.model)

        # setup scheduler
        self.scheduler = self.init_scheduler(self.config, self.optimizer)

        if self.args.restore_path:
            self.model, self.optimizer, self.scaler, self.restore_step = self.restore_model(
                self.config, args.restore_path, self.model, self.optimizer,
                self.scaler)

        if self.use_cuda:
            self.model.cuda()
            self.criterion.cuda()

        # DISTRUBUTED
        if self.num_gpus > 1:
            self.model = DDP_th(self.model, device_ids=[args.rank])

        # count model size
        num_params = count_parameters(self.model)
        logging.info("\n > Model has {} parameters".format(num_params),
                     flush=True)

    def init_model(self):
        model = setup_model(
            len(self.model_characters),
            self.speaker_manager.num_speakers,
            self.config,
            self.speaker_manager.x_vector_dim
            if self.speaker_manager.x_vectors else None,
        )
        return model

    def init_optimizer(self, model):
        optimizer_name = self.config.optimizer
        optimizer_params = self.config.optimizer_params
        if optimizer_name.lower() == "radam":
            module = importlib.import_module("TTS.utils.radam")
            optimizer = getattr(module, "RAdam")
        else:
            optimizer = getattr(torch.optim, optimizer_name)
        return optimizer(model.parameters(),
                         lr=self.config.lr,
                         **optimizer_params)

    def init_character_processor(self):
        # setup custom characters if set in config file.
        # TODO: implement CharacterProcessor
        if self.config.characters is not None:
            symbols, phonemes = make_symbols(
                **self.config.characters.to_dict())
        model_characters = phonemes if self.config.use_phonemes else symbols
        return model_characters

    def init_speaker_manager(self, restore_path: str = "", out_path: str = ""):
        speaker_manager = SpeakerManager()
        if restore_path:
            speakers_file = os.path.join(os.path.dirname(restore_path),
                                         "speaker.json")
            if not os.path.exists(speakers_file):
                logging.info(
                    "WARNING: speakers.json was not found in restore_path, trying to use CONFIG.external_speaker_embedding_file"
                )
                speakers_file = self.config.external_speaker_embedding_file

            if self.config.use_external_speaker_embedding_file:
                speaker_manager.load_x_vectors_file(speakers_file)
            else:
                self.speaker_manage.load_speaker_mapping(speakers_file)
        elif self.config.use_external_speaker_embedding_file and self.config.external_speaker_embedding_file:
            speaker_manager.load_x_vectors_file(
                self.config.external_speaker_embedding_file)
        else:
            speaker_manager.parse_speakers_from_items(self.data_train)
            file_path = os.path.join(out_path, "speakers.json")
            speaker_manager.save_ids_file(file_path)
        return speaker_manager

    def init_scheduler(self, config, optimizer):
        lr_scheduler = config.lr_scheduler
        lr_scheduler_params = config.lr_scheduler_params
        if lr_scheduler.lower() == "noamlr":
            from TTS.utils.training import NoamLR
            scheduler = NoamLR
        else:
            scheduler = getattr(torch.optim, lr_scheduler)
        return scheduler(optimizer, **lr_scheduler_params)

    def init_criterion(self):
        return setup_loss(self.config)

    def restore_model(self,
                      config,
                      restore_path,
                      model,
                      optimizer,
                      scaler=None):
        logging.info(f" > Restoring from {os.path.basename(restore_path)}...")
        checkpoint = torch.load(restore_path, map_location="cpu")
        try:
            logging.info(" > Restoring Model...")
            model.load_state_dict(checkpoint["model"])
            # optimizer restore
            logging.info(" > Restoring Optimizer...")
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scaler" in checkpoint and config.mixed_precision:
                logging.info(" > Restoring AMP Scaler...")
                scaler.load_state_dict(checkpoint["scaler"])
        except (KeyError, RuntimeError):
            logging.info(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint["model"], config)
            model.load_state_dict(model_dict)
            del model_dict

        for group in optimizer.param_groups:
            group["lr"] = self.config.lr
        logging.info(" > Model restored from step %d" % checkpoint["step"],
                     flush=True)
        restore_step = checkpoint["step"]
        return model, optimizer, scaler, restore_step

    def _setup_loader(self, r, ap, is_eval, data_items, verbose,
                      speaker_mapping):
        if is_eval and not self.config.run_eval:
            loader = None
        else:
            dataset = TTSDataset(
                outputs_per_step=r,
                text_cleaner=self.config.text_cleaner,
                compute_linear_spec= 'tacotron' == self.config.model.lower(),
                meta_data=data_items,
                ap=ap,
                tp=self.config.characters,
                add_blank=self.config["add_blank"],
                batch_group_size=0 if is_eval else
                self.config.batch_group_size * self.config.batch_size,
                min_seq_len=self.config.min_seq_len,
                max_seq_len=self.config.max_seq_len,
                phoneme_cache_path=self.config.phoneme_cache_path,
                use_phonemes=self.config.use_phonemes,
                phoneme_language=self.config.phoneme_language,
                enable_eos_bos=self.config.enable_eos_bos_chars,
                use_noise_augment=not is_eval,
                verbose=verbose,
                speaker_mapping=speaker_mapping
                if self.config.use_speaker_embedding
                and self.config.use_external_speaker_embedding_file else None,
            )

            if self.config.use_phonemes and self.config.compute_input_seq_cache:
                # precompute phonemes to have a better estimate of sequence lengths.
                dataset.compute_input_seq(self.config.num_loader_workers)
            dataset.sort_items()

            sampler = DistributedSampler(
                dataset) if self.num_gpus > 1 else None
            loader = DataLoader(
                dataset,
                batch_size=self.config.eval_batch_size
                if is_eval else self.config.batch_size,
                shuffle=False,
                collate_fn=dataset.collate_fn,
                drop_last=False,
                sampler=sampler,
                num_workers=self.config.num_val_loader_workers
                if is_eval else self.config.num_loader_workers,
                pin_memory=False,
            )
        return loader

    def setup_train_dataloader(self, r, ap, data_items, verbose,
                               speaker_mapping):
        return self._setup_loader(r, ap, False, data_items, verbose,
                                  speaker_mapping)

    def setup_eval_dataloder(self, r, ap, data_items, verbose,
                             speaker_mapping):
        return self._setup_loader(r, ap, True, data_items, verbose,
                                  speaker_mapping)

    def format_batch(self, batch):
        # setup input batch
        text_input = batch[0]
        text_lengths = batch[1]
        speaker_names = batch[2]
        linear_input = batch[3] if self.config.model.lower() in ["tacotron"
                                                                 ] else None
        mel_input = batch[4]
        mel_lengths = batch[5]
        stop_targets = batch[6]
        item_idx = batch[7]
        attn_mask = batch[8]
        max_text_length = torch.max(text_lengths.float())
        max_spec_length = torch.max(mel_lengths.float())

        # convert speaker names to ids
        if self.config.use_speaker_embedding:
            if self.config.use_external_speaker_embedding_file:
                speaker_embeddings = batch[8]
                speaker_ids = None
            else:
                speaker_ids = [
                    self.speaker_manager.speaker_ids[speaker_name]
                    for speaker_name in speaker_names
                ]
                speaker_ids = torch.LongTensor(speaker_ids)
                speaker_embeddings = None
        else:
            speaker_embeddings = None
            speaker_ids = None

        # compute durations from attention masks
        if attn_mask:
            durations = torch.zeros(attn_mask.shape[0], attn_mask.shape[2])
            for idx, am in enumerate(attn_mask):
                # compute raw durations
                c_idxs = am[:, :text_lengths[idx], :mel_lengths[idx]].max(1)[1]
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
                durations[idx, :text_lengths[idx]] = dur

        # set stop targets view, we predict a single stop token per iteration.
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // self.config.r,
                                         -1)
        stop_targets = (stop_targets.sum(2) >
                        0.0).unsqueeze(2).float().squeeze(2)

        # dispatch batch to GPU
        if self.use_cuda:
            text_input = text_input.cuda(non_blocking=True)
            text_lengths = text_lengths.cuda(non_blocking=True)
            mel_input = mel_input.cuda(non_blocking=True)
            mel_lengths = mel_lengths.cuda(non_blocking=True)
            linear_input = linear_input.cuda(
                non_blocking=True) if self.config.model.lower() in [
                    "tacotron"
                ] else None
            stop_targets = stop_targets.cuda(non_blocking=True)
            attn_mask = attn_mask.cuda(
                non_blocking=True) if attn_mask else None
            durations = durations.cuda(
                non_blocking=True) if attn_mask else None
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
            "item_idx": item_idx
        }

    def train_step(self, batch, batch_n_steps, step, loader_start_time):
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
            raise RuntimeError(
                f"Detected NaN loss at step {self.total_steps_done}.")

        # optimizer step
        if self.config.mixed_precision:
            # model optimizer step in mixed precision mode
            self.scaler.scale(loss_dict["loss"]).backward()
            self.scaler.unscale_(self.optimizer)
            grad_norm, _ = check_update(self.model,
                                        self.config.grad_clip,
                                        ignore_stopnet=True)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # main model optimizer step
            loss_dict["loss"].backward()
            grad_norm, _ = check_update(self.model,
                                        self.config.grad_clip,
                                        ignore_stopnet=True)
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
                "max_spec_length": [batch["max_spec_length"],
                                    1],  # value, precision
                "max_text_length": [batch["max_text_length"], 1],
                "step_time": [step_time, 4],
                "loader_time": [loader_time, 2],
                "current_lr": current_lr,
            }
            self.c_logger.print_train_step(batch_n_steps, step,
                                           self.total_steps_done, log_dict,
                                           loss_dict,
                                           self.keep_avg_train.avg_values)

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
                self.tb_logger.tb_train_step_stats(self.total_steps_done,
                                                   iter_stats)

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
                        scaler=self.scaler.state_dict()
                        if self.config.mixed_precision else None,
                    )
                # training visualizations
                figures, audios = self.model.train_log(self.ap, batch, outputs)
                self.tb_logger.tb_train_figures(self.total_steps_done, figures)
                self.tb_logger.tb_train_audios(self.total_steps_done,
                                               {"TrainAudio": audios},
                                               self.ap.sample_rate)
        self.total_steps_done += 1
        return outputs, loss_dict

    def train_epoch(self):
        self.model.train()
        epoch_start_time = time.time()
        if self.use_cuda:
            batch_num_steps = int(
                len(self.train_loader.dataset) /
                (self.config.batch_size * self.num_gpus))
        else:
            batch_num_steps = int(
                len(self.train_loader.dataset) / self.config.batch_size)
        self.c_logger.print_train_start()
        loader_start_time = time.time()
        for cur_step, batch in enumerate(self.train_loader):
            _, _ = self.train_step(batch, batch_num_steps, cur_step,
                                   loader_start_time)
        epoch_time = time.time() - epoch_start_time
        # Plot self.epochs_done Stats
        if self.args.rank == 0:
            epoch_stats = {"epoch_time": epoch_time}
            epoch_stats.update(self.keep_avg_train.avg_values)
            self.tb_logger.tb_train_epoch_stats(self.total_steps_done,
                                                epoch_stats)
            if self.config.tb_model_param_stats:
                self.tb_logger.tb_model_weights(self.model,
                                                self.total_steps_done)

    def eval_step(self, batch, step):
        with torch.no_grad():
            step_start_time = time.time()

            with torch.cuda.amp.autocast(enabled=self.config.mixed_precision):
                outputs, loss_dict = self.model.eval_step(
                    batch, self.criterion)

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
                self.c_logger.print_eval_step(step, loss_dict,
                                              self.keep_eval_avg.avg_values)
        return outputs, loss_dict

    def eval_epoch(self):
        self.model.eval()
        if self.use_cuda:
            batch_num_steps = int(
                len(self.train_loader.dataset) /
                (self.config.batch_size * self.num_gpus))
        else:
            batch_num_steps = int(
                len(self.train_loader.dataset) / self.config.batch_size)
        self.c_logger.print_eval_start()
        loader_start_time = time.time()
        for cur_step, batch in enumerate(self.eval_loader):
            # format data
            batch = self.format_batch(batch)
            loader_time = time.time() - loader_start_time
            self.keep_avg_eval.update_values({'avg_loader_time': loader_time})
            outputs, _ = self.eval_step(batch, cur_step)
        # Plot epoch stats and samples from the last batch.
        if self.args.rank == 0:
            figures, eval_audios = self.model.eval_log(self.ap, batch, outputs)
            self.tb_logger.tb_eval_figures(self.total_steps_done, figures)
            self.tb_logger.tb_eval_audios(self.total_steps_done,
                                          {"EvalAudio": eval_audios},
                                          self.ap.sample_rate)

    def test_run(self, ):
        logging.info(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        cond_inputs = self._get_cond_inputs()
        for idx, sen in enumerate(test_sentences):
            wav, alignment, decoder_output, postnet_output, stop_tokens, _ = synthesis(
                self.model,
                sen,
                self.config,
                self.use_cuda,
                self.ap,
                speaker_id=cond_inputs['speaker_id'],
                x_vector=cond_inputs['x_vector'],
                style_wav=cond_inputs['style_wav'],
                enable_eos_bos_chars=self.config.enable_eos_bos_chars,
                use_griffin_lim=True,
                do_trim_silence=False,
            )

            file_path = os.path.join(self.output_audio_path, str(self.total_steps_done))
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path, "TestSentence_{}.wav".format(idx))
            self.ap.save_wav(wav, file_path)
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                postnet_output, self.ap, output_fig=False)
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                alignment, output_fig=False)

        self.tb_logger.tb_test_audios(self.total_steps_done, test_audios,
                                 self.config.audio["sample_rate"])
        self.tb_logger.tb_test_figures(self.total_steps_done, test_figures)

    def _get_cond_inputs(self):
        # setup speaker_id
        speaker_id = 0 if self.config.use_speaker_embedding else None
        # setup x_vector
        x_vector = self.speaker_manager.get_x_vectors_by_speaker(
            self.speaker_manager.speaker_ids[0]
        ) if self.config.use_external_speaker_embedding_file and self.config.use_speaker_embedding else None
        # setup style_mel
        style_wav = self.config.gst_style_input
        if style_wav is None and self.config.use_gst:
            # inicialize GST with zero dict.
            style_wav = {}
            print("WARNING: You don't provided a gst style wav, for this reason we use a zero tensor!")
            for i in range(self.config.gst["gst_num_style_tokens"]):
                style_wav[str(i)] = 0
        cond_inputs = {'speaker_id': speaker_id, 'style_wav': style_wav, 'x_vector': x_vector}
        return cond_inputs

    def fit(self):
        if self.restore_step != 0 or self.args.best_path:
            logging.info(" > Restoring best loss from "
                         f"{os.path.basename(self.args.best_path)} ...")
            self.best_loss = torch.load(self.args.best_path,
                                        map_location="cpu")["model_loss"]
            logging.info(
                f" > Starting with loaded last best loss {self.best_loss}.")

        # define data loaders
        self.train_loader = self.setup_train_dataloader(
            self.config.r,
            self.ap,
            self.data_train,
            verbose=True,
            speaker_mapping=self.speaker_manager.speaker_ids)
        self.eval_loader = self.setup_eval_dataloder(
            self.config.r,
            self.ap,
            self.data_train,
            verbose=True,
            speaker_mapping=self.speaker_manager.speaker_ids
        ) if self.config.run_eval else None

        self.total_steps_done = self.restore_step

        for epoch in range(0, self.config.epochs):
            self.keep_avg_train = KeepAverage()
            self.keep_avg_eval = KeepAverage(
            ) if self.config.run_eval else None
            self.epochs_done = epoch
            self.on_epoch_start()
            self.c_logger.print_epoch_start(epoch, self.config.epochs)
            self.train_epoch()
            if self.config.run_eval:
                self.eval_epoch()
            self.test_run()
            self.on_epoch_end()
            self.c_logger.print_epoch_end(
                epoch, self.keep_avg_eval.avg_values
                if self.config.run_eval else self.keep_avg_train.avg_values)
            self.save_best_model()

    def save_best_model(self):
        self.best_loss = save_best_model(
            self.keep_avg_eval['avg_loss']
            if self.keep_avg_eval else self.keep_avg_train['avg_loss'],
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
            scaler=self.scaler.state_dict()
            if self.config.mixed_precision else None,
        )

    def on_epoch_start(self):
        self.model.on_epoch_start(self)

    def on_epoch_end(self):
        ...

    def on_step_start(self):
        ...

    def on_step_end(self):
        ...
