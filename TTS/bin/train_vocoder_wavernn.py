#!/usr/bin/env python3
"""Train WaveRNN vocoder model."""

import os
import sys
import traceback
import time
import random

import torch
from torch.utils.data import DataLoader

# from torch.utils.data.distributed import DistributedSampler

from TTS.utils.arguments import parse_arguments, process_args
from TTS.tts.utils.visual import plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.utils.radam import RAdam
from TTS.utils.training import setup_torch_training_env
from TTS.utils.generic_utils import (
    KeepAverage,
    count_parameters,
    remove_experiment_folder,
    set_init_dict,
)
from TTS.vocoder.datasets.wavernn_dataset import WaveRNNDataset
from TTS.vocoder.datasets.preprocess import (
    load_wav_data,
    load_wav_feat_data
)
from TTS.vocoder.utils.distribution import discretized_mix_logistic_loss, gaussian_loss
from TTS.vocoder.utils.generic_utils import setup_generator
from TTS.vocoder.utils.io import save_best_model, save_checkpoint


use_cuda, num_gpus = setup_torch_training_env(True, True)


def setup_loader(ap, is_val=False, verbose=False):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = WaveRNNDataset(ap=ap,
                                 items=eval_data if is_val else train_data,
                                 seq_len=c.seq_len,
                                 hop_len=ap.hop_length,
                                 pad=c.padding,
                                 mode=c.mode,
                                 mulaw=c.mulaw,
                                 is_training=not is_val,
                                 verbose=verbose,
                                 )
        # sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(dataset,
                            shuffle=True,
                            collate_fn=dataset.collate,
                            batch_size=c.batch_size,
                            num_workers=c.num_val_loader_workers
                            if is_val
                            else c.num_loader_workers,
                            pin_memory=True,
                            )
    return loader


def format_data(data):
    # setup input data
    x_input = data[0]
    mels = data[1]
    y_coarse = data[2]

    # dispatch data to GPU
    if use_cuda:
        x_input = x_input.cuda(non_blocking=True)
        mels = mels.cuda(non_blocking=True)
        y_coarse = y_coarse.cuda(non_blocking=True)

    return x_input, mels, y_coarse


def train(model, optimizer, criterion, scheduler, scaler, ap, global_step, epoch):
    # create train loader
    data_loader = setup_loader(ap, is_val=False, verbose=(epoch == 0))
    model.train()
    epoch_time = 0
    keep_avg = KeepAverage()
    if use_cuda:
        batch_n_iter = int(len(data_loader.dataset) /
                           (c.batch_size * num_gpus))
    else:
        batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    end_time = time.time()
    c_logger.print_train_start()
    # train loop
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()
        x_input, mels, y_coarse = format_data(data)
        loader_time = time.time() - end_time
        global_step += 1

        optimizer.zero_grad()

        if c.mixed_precision:
            # mixed precision training
            with torch.cuda.amp.autocast():
                y_hat = model(x_input, mels)
                if isinstance(model.mode, int):
                    y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
                else:
                    y_coarse = y_coarse.float()
                y_coarse = y_coarse.unsqueeze(-1)
                # compute losses
                loss = criterion(y_hat, y_coarse)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            if c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            # full precision training
            y_hat = model(x_input, mels)
            if isinstance(model.mode, int):
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            else:
                y_coarse = y_coarse.float()
            y_coarse = y_coarse.unsqueeze(-1)
            # compute losses
            loss = criterion(y_hat, y_coarse)
            if loss.item() is None:
                raise RuntimeError(" [!] None loss. Exiting ...")
            loss.backward()
            if c.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), c.grad_clip)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # get the current learning rate
        cur_lr = list(optimizer.param_groups)[0]["lr"]

        step_time = time.time() - start_time
        epoch_time += step_time

        update_train_values = dict()
        loss_dict = dict()
        loss_dict["model_loss"] = loss.item()
        for key, value in loss_dict.items():
            update_train_values["avg_" + key] = value
        update_train_values["avg_loader_time"] = loader_time
        update_train_values["avg_step_time"] = step_time
        keep_avg.update_values(update_train_values)

        # print training stats
        if global_step % c.print_step == 0:
            log_dict = {"step_time": [step_time, 2],
                        "loader_time": [loader_time, 4],
                        "current_lr": cur_lr,
                        }
            c_logger.print_train_step(batch_n_iter,
                                      num_iter,
                                      global_step,
                                      log_dict,
                                      loss_dict,
                                      keep_avg.avg_values,
                                      )

        # plot step stats
        if global_step % 10 == 0:
            iter_stats = {"lr": cur_lr, "step_time": step_time}
            iter_stats.update(loss_dict)
            tb_logger.tb_train_iter_stats(global_step, iter_stats)

        # save checkpoint
        if global_step % c.save_step == 0:
            if c.checkpoint:
                # save model
                save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    None,
                    None,
                    None,
                    global_step,
                    epoch,
                    OUT_PATH,
                    model_losses=loss_dict,
                    scaler=scaler.state_dict() if c.mixed_precision else None
                )

            # synthesize a full voice
            rand_idx = random.randrange(0, len(train_data))
            wav_path = train_data[rand_idx] if not isinstance(
                train_data[rand_idx], (tuple, list)) else train_data[rand_idx][0]
            wav = ap.load_wav(wav_path)
            ground_mel = ap.melspectrogram(wav)
            ground_mel = torch.FloatTensor(ground_mel)
            if use_cuda:
                ground_mel = ground_mel.cuda(non_blocking=True)
            sample_wav = model.inference(ground_mel,
                                         c.batched,
                                         c.target_samples,
                                         c.overlap_samples,
                                         )
            predict_mel = ap.melspectrogram(sample_wav)

            # compute spectrograms
            figures = {"train/ground_truth": plot_spectrogram(ground_mel.T),
                       "train/prediction": plot_spectrogram(predict_mel.T)
                       }
            tb_logger.tb_train_figures(global_step, figures)

            # Sample audio
            tb_logger.tb_train_audios(
                global_step, {
                    "train/audio": sample_wav}, c.audio["sample_rate"]
            )
        end_time = time.time()

    # print epoch stats
    c_logger.print_train_epoch_end(global_step, epoch, epoch_time, keep_avg)

    # Plot Training Epoch Stats
    epoch_stats = {"epoch_time": epoch_time}
    epoch_stats.update(keep_avg.avg_values)
    tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
    # TODO: plot model stats
    # if c.tb_model_param_stats:
    # tb_logger.tb_model_weights(model, global_step)
    return keep_avg.avg_values, global_step


@torch.no_grad()
def evaluate(model, criterion, ap, global_step, epoch):
    # create train loader
    data_loader = setup_loader(ap, is_val=True, verbose=(epoch == 0))
    model.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    end_time = time.time()
    c_logger.print_eval_start()
    with torch.no_grad():
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()
            # format data
            x_input, mels, y_coarse = format_data(data)
            loader_time = time.time() - end_time
            global_step += 1

            y_hat = model(x_input, mels)
            if isinstance(model.mode, int):
                y_hat = y_hat.transpose(1, 2).unsqueeze(-1)
            else:
                y_coarse = y_coarse.float()
            y_coarse = y_coarse.unsqueeze(-1)
            loss = criterion(y_hat, y_coarse)
            # Compute avg loss
            # if num_gpus > 1:
            #     loss = reduce_tensor(loss.data, num_gpus)
            loss_dict = dict()
            loss_dict["model_loss"] = loss.item()

            step_time = time.time() - start_time
            epoch_time += step_time

            # update avg stats
            update_eval_values = dict()
            for key, value in loss_dict.items():
                update_eval_values["avg_" + key] = value
            update_eval_values["avg_loader_time"] = loader_time
            update_eval_values["avg_step_time"] = step_time
            keep_avg.update_values(update_eval_values)

            # print eval stats
            if c.print_eval:
                c_logger.print_eval_step(
                    num_iter, loss_dict, keep_avg.avg_values)

    if epoch % c.test_every_epochs == 0 and epoch != 0:
        # synthesize a full voice
        rand_idx = random.randrange(0, len(eval_data))
        wav_path = eval_data[rand_idx] if not isinstance(
            eval_data[rand_idx], (tuple, list)) else eval_data[rand_idx][0]
        wav = ap.load_wav(wav_path)
        ground_mel = ap.melspectrogram(wav)
        ground_mel = torch.FloatTensor(ground_mel)
        if use_cuda:
            ground_mel = ground_mel.cuda(non_blocking=True)
        sample_wav = model.inference(ground_mel,
                                     c.batched,
                                     c.target_samples,
                                     c.overlap_samples,
                                     )
        predict_mel = ap.melspectrogram(sample_wav)

        # Sample audio
        tb_logger.tb_eval_audios(
            global_step, {
                "eval/audio": sample_wav}, c.audio["sample_rate"]
        )

        # compute spectrograms
        figures = {
            "eval/ground_truth": plot_spectrogram(ground_mel.T),
            "eval/prediction": plot_spectrogram(predict_mel.T)
        }
        tb_logger.tb_eval_figures(global_step, figures)

    tb_logger.tb_eval_stats(global_step, keep_avg.avg_values)
    return keep_avg.avg_values


# FIXME: move args definition/parsing inside of main?
def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global train_data, eval_data

    # setup audio processor
    ap = AudioProcessor(**c.audio)

    # print(f" > Loading wavs from: {c.data_path}")
    # if c.feature_path is not None:
    #     print(f" > Loading features from: {c.feature_path}")
    #     eval_data, train_data = load_wav_feat_data(
    #         c.data_path, c.feature_path, c.eval_split_size
    #     )
    # else:
    #     mel_feat_path = os.path.join(OUT_PATH, "mel")
    #     feat_data = find_feat_files(mel_feat_path)
    #     if feat_data:
    #         print(f" > Loading features from: {mel_feat_path}")
    #         eval_data, train_data = load_wav_feat_data(
    #             c.data_path, mel_feat_path, c.eval_split_size
    #         )
    #     else:
    #         print(" > No feature data found. Preprocessing...")
    #         # preprocessing feature data from given wav files
    #         preprocess_wav_files(OUT_PATH, CONFIG, ap)
    #         eval_data, train_data = load_wav_feat_data(
    #             c.data_path, mel_feat_path, c.eval_split_size
    #         )

    print(f" > Loading wavs from: {c.data_path}")
    if c.feature_path is not None:
        print(f" > Loading features from: {c.feature_path}")
        eval_data, train_data = load_wav_feat_data(
            c.data_path, c.feature_path, c.eval_split_size)
    else:
        eval_data, train_data = load_wav_data(
            c.data_path, c.eval_split_size)
    # setup model
    model_wavernn = setup_generator(c)

    # setup amp scaler
    scaler = torch.cuda.amp.GradScaler() if c.mixed_precision else None

    # define train functions
    if c.mode == "mold":
        criterion = discretized_mix_logistic_loss
    elif c.mode == "gauss":
        criterion = gaussian_loss
    elif isinstance(c.mode, int):
        criterion = torch.nn.CrossEntropyLoss()

    if use_cuda:
        model_wavernn.cuda()
        if isinstance(c.mode, int):
            criterion.cuda()

    optimizer = RAdam(model_wavernn.parameters(), lr=c.lr, weight_decay=0)

    scheduler = None
    if "lr_scheduler" in c:
        scheduler = getattr(torch.optim.lr_scheduler, c.lr_scheduler)
        scheduler = scheduler(optimizer, **c.lr_scheduler_params)
    # slow start for the first 5 epochs
    # lr_lambda = lambda epoch: min(epoch / c.warmup_steps, 1)
    # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # restore any checkpoint
    if args.restore_path:
        print(f" > Restoring from {os.path.basename(args.restore_path)}...")
        checkpoint = torch.load(args.restore_path, map_location="cpu")
        try:
            print(" > Restoring Model...")
            model_wavernn.load_state_dict(checkpoint["model"])
            print(" > Restoring Optimizer...")
            optimizer.load_state_dict(checkpoint["optimizer"])
            if "scheduler" in checkpoint:
                print(" > Restoring Generator LR Scheduler...")
                scheduler.load_state_dict(checkpoint["scheduler"])
                scheduler.optimizer = optimizer
            if "scaler" in checkpoint and c.mixed_precision:
                print(" > Restoring AMP Scaler...")
                scaler.load_state_dict(checkpoint["scaler"])
        except RuntimeError:
            # retore only matching layers.
            print(" > Partial model initialization...")
            model_dict = model_wavernn.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint["model"], c)
            model_wavernn.load_state_dict(model_dict)

        print(" > Model restored from step %d" %
              checkpoint["step"], flush=True)
        args.restore_step = checkpoint["step"]
    else:
        args.restore_step = 0

    # DISTRIBUTED
    # if num_gpus > 1:
    #     model = apply_gradient_allreduce(model)

    num_parameters = count_parameters(model_wavernn)
    print(" > Model has {} parameters".format(num_parameters), flush=True)

    if args.restore_step == 0 or not args.best_path:
        best_loss = float('inf')
        print(" > Starting with inf best loss.")
    else:
        print(" > Restoring best loss from "
              f"{os.path.basename(args.best_path)} ...")
        best_loss = torch.load(args.best_path,
                               map_location='cpu')['model_loss']
        print(f" > Starting with loaded last best loss {best_loss}.")
    keep_all_best = c.get('keep_all_best', False)
    keep_after = c.get('keep_after', 10000)  # void if keep_all_best False

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        c_logger.print_epoch_start(epoch, c.epochs)
        _, global_step = train(model_wavernn, optimizer,
                               criterion, scheduler, scaler, ap, global_step, epoch)
        eval_avg_loss_dict = evaluate(
            model_wavernn, criterion, ap, global_step, epoch)
        c_logger.print_epoch_end(epoch, eval_avg_loss_dict)
        target_loss = eval_avg_loss_dict["avg_model_loss"]
        best_loss = save_best_model(
            target_loss,
            best_loss,
            model_wavernn,
            optimizer,
            scheduler,
            None,
            None,
            None,
            global_step,
            epoch,
            OUT_PATH,
            keep_all_best=keep_all_best,
            keep_after=keep_after,
            model_losses=eval_avg_loss_dict,
            scaler=scaler.state_dict() if c.mixed_precision else None
        )


if __name__ == "__main__":
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
