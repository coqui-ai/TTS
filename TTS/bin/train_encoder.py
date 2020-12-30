#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
import time
import traceback

import torch
from torch.utils.data import DataLoader
from TTS.speaker_encoder.dataset import MyDataset
from TTS.speaker_encoder.losses import AngleProtoLoss, GE2ELoss
from TTS.speaker_encoder.model import SpeakerEncoder
from TTS.speaker_encoder.utils.generic_utils import \
    check_config_speaker_encoder, save_best_model
from TTS.speaker_encoder.utils.visual import plot_embeddings
from TTS.tts.datasets.preprocess import load_meta_data
from TTS.utils.audio import AudioProcessor
from TTS.utils.generic_utils import (count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     remove_experiment_folder, set_init_dict)
from TTS.utils.io import copy_model_files, load_config
from TTS.utils.radam import RAdam
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import NoamLR, check_update

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.manual_seed(54321)
use_cuda = torch.cuda.is_available()
num_gpus = torch.cuda.device_count()
print(" > Using CUDA: ", use_cuda)
print(" > Number of GPUs: ", num_gpus)


def setup_loader(ap: AudioProcessor, is_val: bool=False, verbose: bool=False):
    if is_val:
        loader = None
    else:
        dataset = MyDataset(ap,
                            meta_data_eval if is_val else meta_data_train,
                            voice_len=1.6,
                            num_utter_per_speaker=c.num_utters_per_speaker,
                            num_speakers_in_batch=c.num_speakers_in_batch,
                            skip_speakers=False,
                            storage_size=c.storage["storage_size"],
                            sample_from_storage_p=c.storage["sample_from_storage_p"],
                            additive_noise=c.storage["additive_noise"],
                            verbose=verbose)
        # sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(dataset,
                            batch_size=c.num_speakers_in_batch,
                            shuffle=False,
                            num_workers=c.num_loader_workers,
                            collate_fn=dataset.collate_fn)
    return loader


def train(model, criterion, optimizer, scheduler, ap, global_step):
    data_loader = setup_loader(ap, is_val=False, verbose=True)
    model.train()
    epoch_time = 0
    best_loss = float('inf')
    avg_loss = 0
    avg_loader_time = 0
    end_time = time.time()
    for _, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        inputs = data[0]
        loader_time = time.time() - end_time
        global_step += 1

        # setup lr
        if c.lr_decay:
            scheduler.step()
        optimizer.zero_grad()

        # dispatch data to GPU
        if use_cuda:
            inputs = inputs.cuda(non_blocking=True)
            # labels = labels.cuda(non_blocking=True)

        # forward pass model
        outputs = model(inputs)

        # loss computation
        loss = criterion(
            outputs.view(c.num_speakers_in_batch,
                         outputs.shape[0] // c.num_speakers_in_batch, -1))
        loss.backward()
        grad_norm, _ = check_update(model, c.grad_clip)
        optimizer.step()

        step_time = time.time() - start_time
        epoch_time += step_time

        # Averaged Loss and Averaged Loader Time
        avg_loss = 0.01 * loss.item() \
                   + 0.99 * avg_loss if avg_loss != 0 else loss.item()
        avg_loader_time = 1/c.num_loader_workers * loader_time + \
                          (c.num_loader_workers-1) / c.num_loader_workers * avg_loader_time if avg_loader_time != 0 else loader_time
        current_lr = optimizer.param_groups[0]['lr']

        if global_step % c.steps_plot_stats == 0:
            # Plot Training Epoch Stats
            train_stats = {
                "loss": avg_loss,
                "lr": current_lr,
                "grad_norm": grad_norm,
                "step_time": step_time,
                "avg_loader_time": avg_loader_time
            }
            tb_logger.tb_train_epoch_stats(global_step, train_stats)
            figures = {
                # FIXME: not constant
                "UMAP Plot": plot_embeddings(outputs.detach().cpu().numpy(),
                                             10),
            }
            tb_logger.tb_train_figures(global_step, figures)

        if global_step % c.print_step == 0:
            print(
                "   | > Step:{}  Loss:{:.5f}  AvgLoss:{:.5f}  GradNorm:{:.5f}  "
                "StepTime:{:.2f}  LoaderTime:{:.2f}  AvGLoaderTime:{:.2f}  LR:{:.6f}".format(
                    global_step, loss.item(), avg_loss, grad_norm, step_time,
                    loader_time, avg_loader_time, current_lr),
                flush=True)

        # save best model
        best_loss = save_best_model(model, optimizer, avg_loss, best_loss,
                                    OUT_PATH, global_step)

        end_time = time.time()
    return avg_loss, global_step


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global meta_data_train
    global meta_data_eval

    ap = AudioProcessor(**c.audio)
    model = SpeakerEncoder(input_dim=c.model['input_dim'],
                           proj_dim=c.model['proj_dim'],
                           lstm_dim=c.model['lstm_dim'],
                           num_lstm_layers=c.model['num_lstm_layers'])
    optimizer = RAdam(model.parameters(), lr=c.lr)

    if c.loss == "ge2e":
        criterion = GE2ELoss(loss_method='softmax')
    elif c.loss == "angleproto":
        criterion = AngleProtoLoss()
    else:
        raise Exception("The %s  not is a loss supported" % c.loss)

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        try:
            # TODO: fix optimizer init, model.cuda() needs to be called before
            # optimizer restore
            # optimizer.load_state_dict(checkpoint['optimizer'])
            if c.reinit_layers:
                raise RuntimeError
            model.load_state_dict(checkpoint['model'])
        except KeyError:
            print(" > Partial model initialization.")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint, c)
            model.load_state_dict(model_dict)
            del model_dict
        for group in optimizer.param_groups:
            group['lr'] = c.lr
        print(" > Model restored from step %d" % checkpoint['step'],
              flush=True)
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0

    if use_cuda:
        model = model.cuda()
        criterion.cuda()

    if c.lr_decay:
        scheduler = NoamLR(optimizer,
                           warmup_steps=c.warmup_steps,
                           last_epoch=args.restore_step - 1)
    else:
        scheduler = None

    num_params = count_parameters(model)
    print("\n > Model has {} parameters".format(num_params), flush=True)

    # pylint: disable=redefined-outer-name
    meta_data_train, meta_data_eval = load_meta_data(c.datasets)

    global_step = args.restore_step
    _, global_step = train(model, criterion, optimizer, scheduler, ap,
                           global_step)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Path to model outputs (checkpoint, tensorboard etc.).',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        required=True,
        help='Path to config file for training.',
    )
    parser.add_argument('--debug',
                        type=bool,
                        default=True,
                        help='Do not verify commit integrity to run training.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='Defines the data path. It overwrites config.json.')
    parser.add_argument('--output_path',
                        type=str,
                        help='path for training outputs.',
                        default='')
    parser.add_argument('--output_folder',
                        type=str,
                        default='',
                        help='folder name for training outputs.')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    check_config_speaker_encoder(c)
    _ = os.path.dirname(os.path.realpath(__file__))
    if args.data_path != '':
        c.data_path = args.data_path

    if args.output_path == '':
        OUT_PATH = os.path.join(_, c.output_path)
    else:
        OUT_PATH = args.output_path

    if args.output_folder == '':
        OUT_PATH = create_experiment_folder(OUT_PATH, c.run_name, args.debug)
    else:
        OUT_PATH = os.path.join(OUT_PATH, args.output_folder)

    new_fields = {}
    if args.restore_path:
        new_fields["restore_path"] = args.restore_path
    new_fields["github_branch"] = get_git_branch()
    copy_model_files(c,  args.config_path, OUT_PATH,
                     new_fields)

    LOG_DIR = OUT_PATH
    tb_logger = TensorboardLogger(LOG_DIR, model_name='Speaker_Encoder')

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
