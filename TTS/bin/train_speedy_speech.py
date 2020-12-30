#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import glob
import os
import sys
import time
import traceback
import numpy as np
from random import randrange

import torch
# DISTRIBUTED
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.tts.datasets.TTSDataset import MyDataset
from TTS.tts.layers.losses import SpeedySpeechLoss
from TTS.tts.utils.generic_utils import check_config_tts, setup_model
from TTS.tts.utils.io import save_best_model, save_checkpoint
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import parse_speakers
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.console_logger import ConsoleLogger
from TTS.utils.distribute import init_distributed, reduce_tensor
from TTS.utils.generic_utils import (KeepAverage, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     remove_experiment_folder, set_init_dict)
from TTS.utils.io import copy_model_files, load_config
from TTS.utils.radam import RAdam
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import NoamLR, setup_torch_training_env

use_cuda, num_gpus = setup_torch_training_env(True, False)


def setup_loader(ap, r, is_val=False, verbose=False):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = MyDataset(
            r,
            c.text_cleaner,
            compute_linear_spec=False,
            meta_data=meta_data_eval if is_val else meta_data_train,
            ap=ap,
            tp=c.characters if 'characters' in c.keys() else None,
            add_blank=c['add_blank'] if 'add_blank' in c.keys() else False,
            batch_group_size=0 if is_val else c.batch_group_size *
            c.batch_size,
            min_seq_len=c.min_seq_len,
            max_seq_len=c.max_seq_len,
            phoneme_cache_path=c.phoneme_cache_path,
            use_phonemes=c.use_phonemes,
            phoneme_language=c.phoneme_language,
            enable_eos_bos=c.enable_eos_bos_chars,
            use_noise_augment=not is_val,
            verbose=verbose,
            speaker_mapping=speaker_mapping if c.use_speaker_embedding and c.use_external_speaker_embedding_file else None)

        if c.use_phonemes and c.compute_input_seq_cache:
            # precompute phonemes to have a better estimate of sequence lengths.
            dataset.compute_input_seq(c.num_loader_workers)
        dataset.sort_items()

        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(
            dataset,
            batch_size=c.eval_batch_size if is_val else c.batch_size,
            shuffle=False,
            collate_fn=dataset.collate_fn,
            drop_last=False,
            sampler=sampler,
            num_workers=c.num_val_loader_workers
            if is_val else c.num_loader_workers,
            pin_memory=False)
    return loader


def format_data(data):
    # setup input data
    text_input = data[0]
    text_lengths = data[1]
    speaker_names = data[2]
    mel_input = data[4].permute(0, 2, 1)  # B x D x T
    mel_lengths = data[5]
    item_idx = data[7]
    attn_mask = data[9]
    avg_text_length = torch.mean(text_lengths.float())
    avg_spec_length = torch.mean(mel_lengths.float())

    if c.use_speaker_embedding:
        if c.use_external_speaker_embedding_file:
            # return precomputed embedding vector
            speaker_c = data[8]
        else:
            # return speaker_id to be used by an embedding layer
            speaker_c = [
                speaker_mapping[speaker_name] for speaker_name in speaker_names
            ]
            speaker_c = torch.LongTensor(speaker_c)
    else:
        speaker_c = None
    # compute durations from attention mask
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
        assert dur.sum() == mel_lengths[idx], f" [!] total duration {dur.sum()} vs spectrogram length {mel_lengths[idx]}"
        durations[idx, :text_lengths[idx]] = dur
    # dispatch data to GPU
    if use_cuda:
        text_input = text_input.cuda(non_blocking=True)
        text_lengths = text_lengths.cuda(non_blocking=True)
        mel_input = mel_input.cuda(non_blocking=True)
        mel_lengths = mel_lengths.cuda(non_blocking=True)
        if speaker_c is not None:
            speaker_c = speaker_c.cuda(non_blocking=True)
        attn_mask = attn_mask.cuda(non_blocking=True)
        durations = durations.cuda(non_blocking=True)
    return text_input, text_lengths, mel_input, mel_lengths, speaker_c,\
         avg_text_length, avg_spec_length, attn_mask, durations, item_idx


def train(data_loader, model, criterion, optimizer, scheduler,
          ap, global_step, epoch):

    model.train()
    epoch_time = 0
    keep_avg = KeepAverage()
    if use_cuda:
        batch_n_iter = int(
            len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    c_logger.print_train_start()
    scaler = torch.cuda.amp.GradScaler() if c.mixed_precision else None
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        text_input, text_lengths, mel_targets, mel_lengths, speaker_c,\
            avg_text_length, avg_spec_length, _, dur_target, _ = format_data(data)

        loader_time = time.time() - end_time

        global_step += 1
        optimizer.zero_grad()

        # forward pass model
        with torch.cuda.amp.autocast(enabled=c.mixed_precision):
            decoder_output, dur_output, alignments = model.forward(
                text_input, text_lengths, mel_lengths, dur_target, g=speaker_c)

            # compute loss
            loss_dict = criterion(decoder_output, mel_targets, mel_lengths, dur_output, torch.log(1 + dur_target), text_lengths)

        # backward pass with loss scaling
        if c.mixed_precision:
            scaler.scale(loss_dict['loss']).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           c.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict['loss'].backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           c.grad_clip)
            optimizer.step()

        # setup lr
        if c.noam_schedule:
            scheduler.step()

        # current_lr
        current_lr = optimizer.param_groups[0]['lr']

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(alignments, binary=True)
        loss_dict['align_error'] = align_error

        step_time = time.time() - start_time
        epoch_time += step_time

        # aggregate losses from processes
        if num_gpus > 1:
            loss_dict['loss_l1'] = reduce_tensor(loss_dict['loss_l1'].data, num_gpus)
            loss_dict['loss_ssim'] = reduce_tensor(loss_dict['loss_ssim'].data, num_gpus)
            loss_dict['loss_dur'] = reduce_tensor(loss_dict['loss_dur'].data, num_gpus)
            loss_dict['loss'] = reduce_tensor(loss_dict['loss'] .data, num_gpus)

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
            update_train_values['avg_' + key] = value
        update_train_values['avg_loader_time'] = loader_time
        update_train_values['avg_step_time'] = step_time
        keep_avg.update_values(update_train_values)

        # print training progress
        if global_step % c.print_step == 0:
            log_dict = {

                "avg_spec_length": [avg_spec_length, 1],  # value, precision
                "avg_text_length": [avg_text_length, 1],
                "step_time": [step_time, 4],
                "loader_time": [loader_time, 2],
                "current_lr": current_lr,
            }
            c_logger.print_train_step(batch_n_iter, num_iter, global_step,
                                      log_dict, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # Plot Training Iter Stats
            # reduce TB load
            if global_step % c.tb_plot_step == 0:
                iter_stats = {
                    "lr": current_lr,
                    "grad_norm": grad_norm,
                    "step_time": step_time
                }
                iter_stats.update(loss_dict)
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer, global_step, epoch, 1, OUT_PATH,
                                    model_loss=loss_dict['loss'])

                # wait all kernels to be completed
                torch.cuda.synchronize()

                # Diagnostic visualizations
                idx = np.random.randint(mel_targets.shape[0])
                pred_spec = decoder_output[idx].detach().data.cpu().numpy().T
                gt_spec = mel_targets[idx].data.cpu().numpy().T
                align_img = alignments[idx].data.cpu()

                figures = {
                    "prediction": plot_spectrogram(pred_spec, ap),
                    "ground_truth": plot_spectrogram(gt_spec, ap),
                    "alignment": plot_alignment(align_img),
                }

                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                train_audio = ap.inv_melspectrogram(pred_spec.T)
                tb_logger.tb_train_audios(global_step,
                                          {'TrainAudio': train_audio},
                                          c.audio["sample_rate"])
        end_time = time.time()

    # print epoch stats
    c_logger.print_train_epoch_end(global_step, epoch, epoch_time, keep_avg)

    # Plot Epoch Stats
    if args.rank == 0:
        epoch_stats = {"epoch_time": epoch_time}
        epoch_stats.update(keep_avg.avg_values)
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
        if c.tb_model_param_stats:
            tb_logger.tb_model_weights(model, global_step)
    return keep_avg.avg_values, global_step


@torch.no_grad()
def evaluate(data_loader, model, criterion, ap, global_step, epoch):
    model.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    c_logger.print_eval_start()
    if data_loader is not None:
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # format data
            text_input, text_lengths, mel_targets, mel_lengths, speaker_c,\
                _, _, _, dur_target, _ = format_data(data)

            # forward pass model
            with torch.cuda.amp.autocast(enabled=c.mixed_precision):
                decoder_output, dur_output, alignments = model.forward(
                    text_input, text_lengths, mel_lengths, dur_target, g=speaker_c)

                # compute loss
                loss_dict = criterion(decoder_output, mel_targets, mel_lengths, dur_output, torch.log(1 + dur_target), text_lengths)

            # step time
            step_time = time.time() - start_time
            epoch_time += step_time

            # compute alignment score
            align_error = 1 - alignment_diagonal_score(alignments, binary=True)
            loss_dict['align_error'] = align_error

            # aggregate losses from processes
            if num_gpus > 1:
                loss_dict['loss_l1'] = reduce_tensor(loss_dict['loss_l1'].data, num_gpus)
                loss_dict['loss_ssim'] = reduce_tensor(loss_dict['loss_ssim'].data, num_gpus)
                loss_dict['loss_dur'] = reduce_tensor(loss_dict['loss_dur'].data, num_gpus)
                loss_dict['loss'] = reduce_tensor(loss_dict['loss'] .data, num_gpus)

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
                update_train_values['avg_' + key] = value
            keep_avg.update_values(update_train_values)

            if c.print_eval:
                c_logger.print_eval_step(num_iter, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # Diagnostic visualizations
            idx = np.random.randint(mel_targets.shape[0])
            pred_spec = decoder_output[idx].detach().data.cpu().numpy().T
            gt_spec = mel_targets[idx].data.cpu().numpy().T
            align_img = alignments[idx].data.cpu()

            eval_figures = {
                "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
                "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
                "alignment": plot_alignment(align_img, output_fig=False)
            }

            # Sample audio
            eval_audio = ap.inv_melspectrogram(pred_spec.T)
            tb_logger.tb_eval_audios(global_step, {"ValAudio": eval_audio},
                                     c.audio["sample_rate"])

            # Plot Validation Stats
            tb_logger.tb_eval_stats(global_step, keep_avg.avg_values)
            tb_logger.tb_eval_figures(global_step, eval_figures)

    if args.rank == 0 and epoch >= c.test_delay_epochs:
        if c.test_sentences_file is None:
            test_sentences = [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "Be a voice, not an echo.",
                "I'm sorry Dave. I'm afraid I can't do that.",
                "This cake is great. It's so delicious and moist.",
                "Prior to November 22, 1963."
            ]
        else:
            with open(c.test_sentences_file, "r") as f:
                test_sentences = [s.strip() for s in f.readlines()]

        # test sentences
        test_audios = {}
        test_figures = {}
        print(" | > Synthesizing test sentences")
        if c.use_speaker_embedding:
            if c.use_external_speaker_embedding_file:
                speaker_embedding = speaker_mapping[list(speaker_mapping.keys())[randrange(len(speaker_mapping)-1)]]['embedding']
                speaker_id = None
            else:
                speaker_id = 0
                speaker_embedding = None
        else:
            speaker_id = None
            speaker_embedding = None

        style_wav = c.get("style_wav_for_test")
        for idx, test_sentence in enumerate(test_sentences):
            try:
                wav, alignment, _, postnet_output, _, _ = synthesis(
                    model,
                    test_sentence,
                    c,
                    use_cuda,
                    ap,
                    speaker_id=speaker_id,
                    speaker_embedding=speaker_embedding,
                    style_wav=style_wav,
                    truncated=False,
                    enable_eos_bos_chars=c.enable_eos_bos_chars, #pylint: disable=unused-argument
                    use_griffin_lim=True,
                    do_trim_silence=False)

                file_path = os.path.join(AUDIO_PATH, str(global_step))
                os.makedirs(file_path, exist_ok=True)
                file_path = os.path.join(file_path,
                                         "TestSentence_{}.wav".format(idx))
                ap.save_wav(wav, file_path)
                test_audios['{}-audio'.format(idx)] = wav
                test_figures['{}-prediction'.format(idx)] = plot_spectrogram(
                    postnet_output, ap)
                test_figures['{}-alignment'.format(idx)] = plot_alignment(
                    alignment)
            except: #pylint: disable=bare-except
                print(" !! Error creating Test Sentence -", idx)
                traceback.print_exc()
        tb_logger.tb_test_audios(global_step, test_audios,
                                 c.audio['sample_rate'])
        tb_logger.tb_test_figures(global_step, test_figures)
    return keep_avg.avg_values


# FIXME: move args definition/parsing inside of main?
def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train, meta_data_eval, symbols, phonemes, speaker_mapping
    # Audio processor
    ap = AudioProcessor(**c.audio)
    if 'characters' in c.keys():
        symbols, phonemes = make_symbols(**c.characters)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])
    num_chars = len(phonemes) if c.use_phonemes else len(symbols)

    # load data instances
    meta_data_train, meta_data_eval = load_meta_data(c.datasets, eval_split=True)

    # set the portion of the data used for training if set in config.json
    if 'train_portion' in c.keys():
        meta_data_train = meta_data_train[:int(len(meta_data_train) * c.train_portion)]
    if 'eval_portion' in c.keys():
        meta_data_eval = meta_data_eval[:int(len(meta_data_eval) * c.eval_portion)]

    # parse speakers
    num_speakers, speaker_embedding_dim, speaker_mapping = parse_speakers(c, args, meta_data_train, OUT_PATH)

    # setup model
    model = setup_model(num_chars, num_speakers, c, speaker_embedding_dim=speaker_embedding_dim)
    optimizer = RAdam(model.parameters(), lr=c.lr, weight_decay=0, betas=(0.9, 0.98), eps=1e-9)
    criterion = SpeedySpeechLoss(c)

    if args.restore_path:
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except: #pylint: disable=bare-except
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            model.load_state_dict(model_dict)
            del model_dict

        for group in optimizer.param_groups:
            group['initial_lr'] = c.lr
        print(" > Model restored from step %d" % checkpoint['step'],
              flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model.cuda()
        criterion.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model = DDP_th(model, device_ids=[args.rank])

    if c.noam_schedule:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.warmup_steps,
                           last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    # define dataloaders
    train_loader = setup_loader(ap, 1, is_val=False, verbose=True)
    eval_loader = setup_loader(ap, 1, is_val=True, verbose=True)

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        c_logger.print_epoch_start(epoch, c.epochs)
        train_avg_loss_dict, global_step = train(train_loader, model, criterion, optimizer,
                                                 scheduler, ap, global_step,
                                                 epoch)
        eval_avg_loss_dict = evaluate(eval_loader , model, criterion, ap, global_step, epoch)
        c_logger.print_epoch_end(epoch, eval_avg_loss_dict)
        target_loss = train_avg_loss_dict['avg_loss']
        if c.run_eval:
            target_loss = eval_avg_loss_dict['avg_loss']
        best_loss = save_best_model(target_loss, best_loss, model, optimizer, global_step, epoch, c.r,
                                    OUT_PATH)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help='Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
        required='--continue_path' not in sys.argv
    )
    parser.add_argument('--debug',
                        type=bool,
                        default=False,
                        help='Do not verify commit integrity to run training.')

    # DISTRUBUTED
    parser.add_argument(
        '--rank',
        type=int,
        default=0,
        help='DISTRIBUTED: process rank for distributed training.')
    parser.add_argument('--group_id',
                        type=str,
                        default="",
                        help='DISTRIBUTED: process group id.')
    args = parser.parse_args()

    if args.continue_path != '':
        args.output_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, 'config.json')
        list_of_files = glob.glob(args.continue_path + "/*.pth.tar") # * means all if need specific format then *.csv
        latest_model_file = max(list_of_files, key=os.path.getctime)
        args.restore_path = latest_model_file
        print(f" > Training continues for {args.restore_path}")

    # setup output paths and read configs
    c = load_config(args.config_path)
    # check_config(c)
    check_config_tts(c)
    _ = os.path.dirname(os.path.realpath(__file__))

    if c.mixed_precision:
        print("   > Mixed precision enabled.")

    OUT_PATH = args.continue_path
    if args.continue_path == '':
        OUT_PATH = create_experiment_folder(c.output_path, c.run_name, args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    c_logger = ConsoleLogger()

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_model_files(c, args.config_path, OUT_PATH, new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

        LOG_DIR = OUT_PATH
        tb_logger = TensorboardLogger(LOG_DIR, model_name='TTS')

        # write model desc to tensorboard
        tb_logger.tb_add_text('model-description', c['run_description'], 0)

    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
