#!/usr/bin/env python3
# TODO: mixed precision training
"""Trains GAN based vocoder model."""

import os
import sys
import time
import traceback
from inspect import signature

import torch
from torch.utils.data import DataLoader
from TTS.utils.arguments import parse_arguments, process_args
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import (KeepAverage, count_parameters,
                                     remove_experiment_folder, set_init_dict)

from TTS.utils.radam import RAdam

from TTS.utils.training import setup_torch_training_env
from TTS.vocoder.datasets.gan_dataset import GANDataset
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data
from TTS.vocoder.layers.losses import DiscriminatorLoss, GeneratorLoss
from TTS.vocoder.utils.generic_utils import (plot_results, setup_discriminator,
                                             setup_generator)
from TTS.vocoder.utils.io import save_best_model, save_checkpoint

# DISTRIBUTED
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.utils.data.distributed import DistributedSampler
from TTS.utils.distribute import init_distributed

use_cuda, num_gpus = setup_torch_training_env(True, True)


def setup_loader(ap, is_val=False, verbose=False):
    loader = None
    if not is_val or c.run_eval:
        dataset = GANDataset(ap=ap,
                             items=eval_data if is_val else train_data,
                             seq_len=c.seq_len,
                             hop_len=ap.hop_length,
                             pad_short=c.pad_short,
                             conv_pad=c.conv_pad,
                             is_training=not is_val,
                             return_segments=not is_val,
                             use_noise_augment=c.use_noise_augment,
                             use_cache=c.use_cache,
                             verbose=verbose)
        dataset.shuffle_mapping()
        sampler = DistributedSampler(dataset, shuffle=True) if num_gpus > 1 else None
        loader = DataLoader(dataset,
                            batch_size=1 if is_val else c.batch_size,
                            shuffle=num_gpus == 0,
                            drop_last=False,
                            sampler=sampler,
                            num_workers=c.num_val_loader_workers
                            if is_val else c.num_loader_workers,
                            pin_memory=False)
    return loader


def format_data(data):
    if isinstance(data[0], list):
        # setup input data
        c_G, x_G = data[0]
        c_D, x_D = data[1]

        # dispatch data to GPU
        if use_cuda:
            c_G = c_G.cuda(non_blocking=True)
            x_G = x_G.cuda(non_blocking=True)
            c_D = c_D.cuda(non_blocking=True)
            x_D = x_D.cuda(non_blocking=True)

        return c_G, x_G, c_D, x_D

    # return a whole audio segment
    co, x = data
    if use_cuda:
        co = co.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)
    return co, x, None, None


def train(model_G, criterion_G, optimizer_G, model_D, criterion_D, optimizer_D,
          scheduler_G, scheduler_D, ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=False, verbose=(epoch == 0))
    model_G.train()
    model_D.train()
    epoch_time = 0
    keep_avg = KeepAverage()
    if use_cuda:
        batch_n_iter = int(
            len(data_loader.dataset) / (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    c_logger.print_train_start()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        c_G, y_G, c_D, y_D = format_data(data)
        loader_time = time.time() - end_time

        global_step += 1

        ##############################
        # GENERATOR
        ##############################

        # generator pass
        y_hat = model_G(c_G)
        y_hat_sub = None
        y_G_sub = None
        y_hat_vis = y_hat  # for visualization

        # PQMF formatting
        if y_hat.shape[1] > 1:
            y_hat_sub = y_hat
            y_hat = model_G.pqmf_synthesis(y_hat)
            y_hat_vis = y_hat
            y_G_sub = model_G.pqmf_analysis(y_G)

        scores_fake, feats_fake, feats_real = None, None, None
        if global_step > c.steps_to_start_discriminator:

            # run D with or without cond. features
            if len(signature(model_D.forward).parameters) == 2:
                D_out_fake = model_D(y_hat, c_G)
            else:
                D_out_fake = model_D(y_hat)
            D_out_real = None

            if c.use_feat_match_loss:
                with torch.no_grad():
                    D_out_real = model_D(y_G)

            # format D outputs
            if isinstance(D_out_fake, tuple):
                scores_fake, feats_fake = D_out_fake
                if D_out_real is None:
                    feats_real = None
                else:
                    _, feats_real = D_out_real
            else:
                scores_fake = D_out_fake

        # compute losses
        loss_G_dict = criterion_G(y_hat, y_G, scores_fake, feats_fake,
                                  feats_real, y_hat_sub, y_G_sub)
        loss_G = loss_G_dict['G_loss']

        # optimizer generator
        optimizer_G.zero_grad()
        loss_G.backward()
        if c.gen_clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model_G.parameters(),
                                           c.gen_clip_grad)
        optimizer_G.step()
        if scheduler_G is not None:
            scheduler_G.step()

        loss_dict = dict()
        for key, value in loss_G_dict.items():
            if isinstance(value, int):
                loss_dict[key] = value
            else:
                loss_dict[key] = value.item()

        ##############################
        # DISCRIMINATOR
        ##############################
        if global_step >= c.steps_to_start_discriminator:
            # discriminator pass
            with torch.no_grad():
                y_hat = model_G(c_D)

            # PQMF formatting
            if y_hat.shape[1] > 1:
                y_hat = model_G.pqmf_synthesis(y_hat)

            # run D with or without cond. features
            if len(signature(model_D.forward).parameters) == 2:
                D_out_fake = model_D(y_hat.detach(), c_D)
                D_out_real = model_D(y_D, c_D)
            else:
                D_out_fake = model_D(y_hat.detach())
                D_out_real = model_D(y_D)

            # format D outputs
            if isinstance(D_out_fake, tuple):
                scores_fake, feats_fake = D_out_fake
                if D_out_real is None:
                    scores_real, feats_real = None, None
                else:
                    scores_real, feats_real = D_out_real
            else:
                scores_fake = D_out_fake
                scores_real = D_out_real

            # compute losses
            loss_D_dict = criterion_D(scores_fake, scores_real)
            loss_D = loss_D_dict['D_loss']

            # optimizer discriminator
            optimizer_D.zero_grad()
            loss_D.backward()
            if c.disc_clip_grad > 0:
                torch.nn.utils.clip_grad_norm_(model_D.parameters(),
                                               c.disc_clip_grad)
            optimizer_D.step()
            if scheduler_D is not None:
                scheduler_D.step()

            for key, value in loss_D_dict.items():
                if isinstance(value, (int, float)):
                    loss_dict[key] = value
                else:
                    loss_dict[key] = value.item()

        step_time = time.time() - start_time
        epoch_time += step_time

        # get current learning rates
        current_lr_G = list(optimizer_G.param_groups)[0]['lr']
        current_lr_D = list(optimizer_D.param_groups)[0]['lr']

        # update avg stats
        update_train_values = dict()
        for key, value in loss_dict.items():
            update_train_values['avg_' + key] = value
        update_train_values['avg_loader_time'] = loader_time
        update_train_values['avg_step_time'] = step_time
        keep_avg.update_values(update_train_values)

        # print training stats
        if global_step % c.print_step == 0:
            log_dict = {
                'step_time': [step_time, 2],
                'loader_time': [loader_time, 4],
                "current_lr_G": current_lr_G,
                "current_lr_D": current_lr_D
            }
            c_logger.print_train_step(batch_n_iter, num_iter, global_step,
                                      log_dict, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # plot step stats
            if global_step % 10 == 0:
                iter_stats = {
                    "lr_G": current_lr_G,
                    "lr_D": current_lr_D,
                    "step_time": step_time
                }
                iter_stats.update(loss_dict)
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            # save checkpoint
            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model_G,
                                    optimizer_G,
                                    scheduler_G,
                                    model_D,
                                    optimizer_D,
                                    scheduler_D,
                                    global_step,
                                    epoch,
                                    OUT_PATH,
                                    model_losses=loss_dict)

                # compute spectrograms
                figures = plot_results(y_hat_vis, y_G, ap, global_step,
                                       'train')
                tb_logger.tb_train_figures(global_step, figures)

                # Sample audio
                sample_voice = y_hat_vis[0].squeeze(0).detach().cpu().numpy()
                tb_logger.tb_train_audios(global_step,
                                          {'train/audio': sample_voice},
                                          c.audio["sample_rate"])
        end_time = time.time()

    # print epoch stats
    c_logger.print_train_epoch_end(global_step, epoch, epoch_time, keep_avg)

    # Plot Training Epoch Stats
    epoch_stats = {"epoch_time": epoch_time}
    epoch_stats.update(keep_avg.avg_values)
    if args.rank == 0:
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
    # TODO: plot model stats
    # if c.tb_model_param_stats:
    # tb_logger.tb_model_weights(model, global_step)
    return keep_avg.avg_values, global_step


@torch.no_grad()
def evaluate(model_G, criterion_G, model_D, criterion_D, ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=True, verbose=(epoch == 0))
    model_G.eval()
    model_D.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    end_time = time.time()
    c_logger.print_eval_start()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        c_G, y_G, _, _ = format_data(data)
        loader_time = time.time() - end_time

        global_step += 1

        ##############################
        # GENERATOR
        ##############################

        # generator pass
        y_hat = model_G(c_G)
        y_hat_sub = None
        y_G_sub = None

        # PQMF formatting
        if y_hat.shape[1] > 1:
            y_hat_sub = y_hat
            y_hat = model_G.pqmf_synthesis(y_hat)
            y_G_sub = model_G.pqmf_analysis(y_G)

        scores_fake, feats_fake, feats_real = None, None, None
        if global_step > c.steps_to_start_discriminator:

            if len(signature(model_D.forward).parameters) == 2:
                D_out_fake = model_D(y_hat, c_G)
            else:
                D_out_fake = model_D(y_hat)
            D_out_real = None

            if c.use_feat_match_loss:
                with torch.no_grad():
                    D_out_real = model_D(y_G)

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
        loss_G_dict = criterion_G(y_hat, y_G, scores_fake, feats_fake,
                                  feats_real, y_hat_sub, y_G_sub)

        loss_dict = dict()
        for key, value in loss_G_dict.items():
            if isinstance(value, (int, float)):
                loss_dict[key] = value
            else:
                loss_dict[key] = value.item()

        ##############################
        # DISCRIMINATOR
        ##############################

        if global_step >= c.steps_to_start_discriminator:
            # discriminator pass
            with torch.no_grad():
                y_hat = model_G(c_G)

            # PQMF formatting
            if y_hat.shape[1] > 1:
                y_hat = model_G.pqmf_synthesis(y_hat)

            # run D with or without cond. features
            if len(signature(model_D.forward).parameters) == 2:
                D_out_fake = model_D(y_hat.detach(), c_G)
                D_out_real = model_D(y_G, c_G)
            else:
                D_out_fake = model_D(y_hat.detach())
                D_out_real = model_D(y_G)

            # format D outputs
            if isinstance(D_out_fake, tuple):
                scores_fake, feats_fake = D_out_fake
                if D_out_real is None:
                    scores_real, feats_real = None, None
                else:
                    scores_real, feats_real = D_out_real
            else:
                scores_fake = D_out_fake
                scores_real = D_out_real

            # compute losses
            loss_D_dict = criterion_D(scores_fake, scores_real)

            for key, value in loss_D_dict.items():
                if isinstance(value, (int, float)):
                    loss_dict[key] = value
                else:
                    loss_dict[key] = value.item()

        step_time = time.time() - start_time
        epoch_time += step_time

        # update avg stats
        update_eval_values = dict()
        for key, value in loss_dict.items():
            update_eval_values['avg_' + key] = value
        update_eval_values['avg_loader_time'] = loader_time
        update_eval_values['avg_step_time'] = step_time
        keep_avg.update_values(update_eval_values)

        # print eval stats
        if c.print_eval:
            c_logger.print_eval_step(num_iter, loss_dict, keep_avg.avg_values)

    if args.rank == 0:
        # compute spectrograms
        figures = plot_results(y_hat, y_G, ap, global_step, 'eval')
        tb_logger.tb_eval_figures(global_step, figures)

        # Sample audio
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        tb_logger.tb_eval_audios(global_step, {'eval/audio': sample_voice},
                                 c.audio["sample_rate"])

        tb_logger.tb_eval_stats(global_step, keep_avg.avg_values)

    # synthesize a full voice
    data_loader.return_segments = False

    return keep_avg.avg_values


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global train_data, eval_data
    print(f" > Loading wavs from: {c.data_path}")
    if c.feature_path is not None:
        print(f" > Loading features from: {c.feature_path}")
        eval_data, train_data = load_wav_feat_data(
            c.data_path, c.feature_path, c.eval_split_size)
    else:
        eval_data, train_data = load_wav_data(c.data_path, c.eval_split_size)

    # setup audio processor
    ap = AudioProcessor(**c.audio)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])

    # setup models
    model_gen = setup_generator(c)
    model_disc = setup_discriminator(c)

    # setup optimizers
    optimizer_gen = RAdam(model_gen.parameters(), lr=c.lr_gen, weight_decay=0)
    optimizer_disc = RAdam(model_disc.parameters(),
                           lr=c.lr_disc,
                           weight_decay=0)

    # schedulers
    scheduler_gen = None
    scheduler_disc = None
    if 'lr_scheduler_gen' in c:
        scheduler_gen = getattr(torch.optim.lr_scheduler, c.lr_scheduler_gen)
        scheduler_gen = scheduler_gen(
            optimizer_gen, **c.lr_scheduler_gen_params)
    if 'lr_scheduler_disc' in c:
        scheduler_disc = getattr(torch.optim.lr_scheduler, c.lr_scheduler_disc)
        scheduler_disc = scheduler_disc(
            optimizer_disc, **c.lr_scheduler_disc_params)

    # setup criterion
    criterion_gen = GeneratorLoss(c)
    criterion_disc = DiscriminatorLoss(c)

    if args.restore_path:
        print(f" > Restoring from {os.path.basename(args.restore_path)}...")
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            print(" > Restoring Generator Model...")
            model_gen.load_state_dict(checkpoint['model'])
            print(" > Restoring Generator Optimizer...")
            optimizer_gen.load_state_dict(checkpoint['optimizer'])
            print(" > Restoring Discriminator Model...")
            model_disc.load_state_dict(checkpoint['model_disc'])
            print(" > Restoring Discriminator Optimizer...")
            optimizer_disc.load_state_dict(checkpoint['optimizer_disc'])
            if 'scheduler' in checkpoint:
                print(" > Restoring Generator LR Scheduler...")
                scheduler_gen.load_state_dict(checkpoint['scheduler'])
                # NOTE: Not sure if necessary
                scheduler_gen.optimizer = optimizer_gen
            if 'scheduler_disc' in checkpoint:
                print(" > Restoring Discriminator LR Scheduler...")
                scheduler_disc.load_state_dict(checkpoint['scheduler_disc'])
                scheduler_disc.optimizer = optimizer_disc
        except RuntimeError:
            # restore only matching layers.
            print(" > Partial model initialization...")
            model_dict = model_gen.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            model_gen.load_state_dict(model_dict)

            model_dict = model_disc.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model_disc'], c)
            model_disc.load_state_dict(model_dict)
            del model_dict

        # reset lr if not countinuining training.
        for group in optimizer_gen.param_groups:
            group['lr'] = c.lr_gen

        for group in optimizer_disc.param_groups:
            group['lr'] = c.lr_disc

        print(f" > Model restored from step {checkpoint['step']:d}",
              flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model_gen.cuda()
        criterion_gen.cuda()
        model_disc.cuda()
        criterion_disc.cuda()

    # DISTRUBUTED
    if num_gpus > 1:
        model_gen = DDP_th(model_gen, device_ids=[args.rank])
        model_disc = DDP_th(model_disc, device_ids=[args.rank])

    num_params = count_parameters(model_gen)
    print(" > Generator has {} parameters".format(num_params), flush=True)
    num_params = count_parameters(model_disc)
    print(" > Discriminator has {} parameters".format(num_params), flush=True)

    if args.restore_step == 0 or not args.best_path:
        best_loss = float('inf')
        print(" > Starting with inf best loss.")
    else:
        print(" > Restoring best loss from "
              f"{os.path.basename(args.best_path)} ...")
        best_loss = torch.load(args.best_path,
                               map_location='cpu')['model_loss']
        print(f" > Starting with best loss of {best_loss}.")
    keep_all_best = c.get('keep_all_best', False)
    keep_after = c.get('keep_after', 10000)  # void if keep_all_best False

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        c_logger.print_epoch_start(epoch, c.epochs)
        _, global_step = train(model_gen, criterion_gen, optimizer_gen,
                               model_disc, criterion_disc, optimizer_disc,
                               scheduler_gen, scheduler_disc, ap, global_step,
                               epoch)
        eval_avg_loss_dict = evaluate(model_gen, criterion_gen, model_disc,
                                      criterion_disc, ap,
                                      global_step, epoch)
        c_logger.print_epoch_end(epoch, eval_avg_loss_dict)
        target_loss = eval_avg_loss_dict[c.target_loss]
        best_loss = save_best_model(target_loss,
                                    best_loss,
                                    model_gen,
                                    optimizer_gen,
                                    scheduler_gen,
                                    model_disc,
                                    optimizer_disc,
                                    scheduler_disc,
                                    global_step,
                                    epoch,
                                    OUT_PATH,
                                    keep_all_best=keep_all_best,
                                    keep_after=keep_after,
                                    model_losses=eval_avg_loss_dict,
                                    )


if __name__ == '__main__':
    args = parse_arguments(sys.argv)
    c, OUT_PATH, AUDIO_PATH, c_logger, tb_logger = process_args(
        args, model_class='vocoder')

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
