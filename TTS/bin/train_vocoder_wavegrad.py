import argparse
import glob
import os
import sys
import time
import traceback
import numpy as np

import torch
# DISTRIBUTED
from torch.nn.parallel import DistributedDataParallel as DDP_th
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from TTS.utils.audio import AudioProcessor
from TTS.utils.console_logger import ConsoleLogger
from TTS.utils.distribute import init_distributed
from TTS.utils.generic_utils import (KeepAverage, count_parameters,
                                     create_experiment_folder, get_git_branch,
                                     remove_experiment_folder, set_init_dict)
from TTS.utils.io import copy_model_files, load_config
from TTS.utils.tensorboard_logger import TensorboardLogger
from TTS.utils.training import setup_torch_training_env
from TTS.vocoder.datasets.preprocess import load_wav_data, load_wav_feat_data
from TTS.vocoder.datasets.wavegrad_dataset import WaveGradDataset
from TTS.vocoder.utils.generic_utils import plot_results, setup_generator
from TTS.vocoder.utils.io import save_best_model, save_checkpoint

use_cuda, num_gpus = setup_torch_training_env(True, True)


def setup_loader(ap, is_val=False, verbose=False):
    if is_val and not c.run_eval:
        loader = None
    else:
        dataset = WaveGradDataset(ap=ap,
                                items=eval_data if is_val else train_data,
                                seq_len=c.seq_len,
                                hop_len=ap.hop_length,
                                pad_short=c.pad_short,
                                conv_pad=c.conv_pad,
                                is_training=not is_val,
                                return_segments=True,
                                use_noise_augment=False,
                                use_cache=c.use_cache,
                                verbose=verbose)
        sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        loader = DataLoader(dataset,
                            batch_size=c.batch_size,
                            shuffle=num_gpus <= 1,
                            drop_last=False,
                            sampler=sampler,
                            num_workers=c.num_val_loader_workers
                            if is_val else c.num_loader_workers,
                            pin_memory=False)


    return loader


def format_data(data):
    # return a whole audio segment
    m, x = data
    x = x.unsqueeze(1)
    if use_cuda:
        m = m.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)
    return m, x


def format_test_data(data):
    # return a whole audio segment
    m, x = data
    m = m[None, ...]
    x = x[None, None, ...]
    if use_cuda:
        m = m.cuda(non_blocking=True)
        x = x.cuda(non_blocking=True)
    return m, x


def train(model, criterion, optimizer,
          scheduler, scaler, ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=False, verbose=(epoch == 0))
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
    # setup noise schedule
    noise_schedule = c['train_noise_schedule']
    betas = np.linspace(noise_schedule['min_val'], noise_schedule['max_val'], noise_schedule['num_steps'])
    if hasattr(model, 'module'):
        model.module.compute_noise_level(betas)
    else:
        model.compute_noise_level(betas)
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        m, x = format_data(data)
        loader_time = time.time() - end_time

        global_step += 1

        with torch.cuda.amp.autocast(enabled=c.mixed_precision):
            # compute noisy input
            if hasattr(model, 'module'):
                noise, x_noisy, noise_scale = model.module.compute_y_n(x)
            else:
                noise, x_noisy, noise_scale = model.compute_y_n(x)

            # forward pass
            noise_hat = model(x_noisy, m, noise_scale)

            # compute losses
            loss = criterion(noise, noise_hat)
        loss_wavegrad_dict = {'wavegrad_loss':loss}

        # check nan loss
        if torch.isnan(loss).any():
            raise RuntimeError(f'Detected NaN loss at step {global_step}.')

        optimizer.zero_grad()

        # backward pass with loss scaling
        if c.mixed_precision:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           c.clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           c.clip_grad)
            optimizer.step()

        # schedule update
        if scheduler is not None:
            scheduler.step()

        # disconnect loss values
        loss_dict = dict()
        for key, value in loss_wavegrad_dict.items():
            if isinstance(value, int):
                loss_dict[key] = value
            else:
                loss_dict[key] = value.item()

        # epoch/step timing
        step_time = time.time() - start_time
        epoch_time += step_time

        # get current learning rates
        current_lr = list(optimizer.param_groups)[0]['lr']

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
                "current_lr": current_lr,
                "grad_norm": grad_norm.item()
            }
            c_logger.print_train_step(batch_n_iter, num_iter, global_step,
                                      log_dict, loss_dict, keep_avg.avg_values)

        if args.rank == 0:
            # plot step stats
            if global_step % 10 == 0:
                iter_stats = {
                    "lr": current_lr,
                    "grad_norm": grad_norm.item(),
                    "step_time": step_time
                }
                iter_stats.update(loss_dict)
                tb_logger.tb_train_iter_stats(global_step, iter_stats)

            # save checkpoint
            if global_step % c.save_step == 0:
                if c.checkpoint:
                    # save model
                    save_checkpoint(model,
                                    optimizer,
                                    scheduler,
                                    None,
                                    None,
                                    None,
                                    global_step,
                                    epoch,
                                    OUT_PATH,
                                    model_losses=loss_dict,
                                    scaler=scaler.state_dict() if c.mixed_precision else None)

        end_time = time.time()

    # print epoch stats
    c_logger.print_train_epoch_end(global_step, epoch, epoch_time, keep_avg)

    # Plot Training Epoch Stats
    epoch_stats = {"epoch_time": epoch_time}
    epoch_stats.update(keep_avg.avg_values)
    if args.rank == 0:
        tb_logger.tb_train_epoch_stats(global_step, epoch_stats)
    # TODO: plot model stats
    if c.tb_model_param_stats and args.rank == 0:
        tb_logger.tb_model_weights(model, global_step)
    return keep_avg.avg_values, global_step


@torch.no_grad()
def evaluate(model, criterion, ap, global_step, epoch):
    data_loader = setup_loader(ap, is_val=True, verbose=(epoch == 0))
    model.eval()
    epoch_time = 0
    keep_avg = KeepAverage()
    end_time = time.time()
    c_logger.print_eval_start()
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # format data
        m, x = format_data(data)
        loader_time = time.time() - end_time

        global_step += 1

        # compute noisy input
        if hasattr(model, 'module'):
            noise, x_noisy, noise_scale = model.module.compute_y_n(x)
        else:
            noise, x_noisy, noise_scale = model.compute_y_n(x)


        # forward pass
        noise_hat = model(x_noisy, m, noise_scale)

        # compute losses
        loss = criterion(noise, noise_hat)
        loss_wavegrad_dict = {'wavegrad_loss':loss}


        loss_dict = dict()
        for key, value in loss_wavegrad_dict.items():
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
        data_loader.dataset.return_segments = False
        samples = data_loader.dataset.load_test_samples(1)
        m, x = format_test_data(samples[0])

        # setup noise schedule and inference
        noise_schedule = c['test_noise_schedule']
        betas = np.linspace(noise_schedule['min_val'], noise_schedule['max_val'], noise_schedule['num_steps'])
        if hasattr(model, 'module'):
            model.module.compute_noise_level(betas)
            # compute voice
            x_pred = model.module.inference(m)
        else:
            model.compute_noise_level(betas)
            # compute voice
            x_pred = model.inference(m)

        # compute spectrograms
        figures = plot_results(x_pred, x, ap, global_step, 'eval')
        tb_logger.tb_eval_figures(global_step, figures)

        # Sample audio
        sample_voice = x_pred[0].squeeze(0).detach().cpu().numpy()
        tb_logger.tb_eval_audios(global_step, {'eval/audio': sample_voice},
                                 c.audio["sample_rate"])

        tb_logger.tb_eval_stats(global_step, keep_avg.avg_values)
        data_loader.dataset.return_segments = True

    return keep_avg.avg_values


def main(args):  # pylint: disable=redefined-outer-name
    # pylint: disable=global-variable-undefined
    global train_data, eval_data
    print(f" > Loading wavs from: {c.data_path}")
    if c.feature_path is not None:
        print(f" > Loading features from: {c.feature_path}")
        eval_data, train_data = load_wav_feat_data(c.data_path, c.feature_path, c.eval_split_size)
    else:
        eval_data, train_data = load_wav_data(c.data_path, c.eval_split_size)

    # setup audio processor
    ap = AudioProcessor(**c.audio)

    # DISTRUBUTED
    if num_gpus > 1:
        init_distributed(args.rank, num_gpus, args.group_id,
                         c.distributed["backend"], c.distributed["url"])

    # setup models
    model = setup_generator(c)

    # scaler for mixed_precision
    scaler = torch.cuda.amp.GradScaler() if c.mixed_precision else None

    # setup optimizers
    optimizer = Adam(model.parameters(), lr=c.lr, weight_decay=0)

    # schedulers
    scheduler = None
    if 'lr_scheduler' in c:
        scheduler = getattr(torch.optim.lr_scheduler, c.lr_scheduler)
        scheduler = scheduler(optimizer, **c.lr_scheduler_params)

    # setup criterion
    criterion = torch.nn.L1Loss().cuda()

    if args.restore_path:
        checkpoint = torch.load(args.restore_path, map_location='cpu')
        try:
            print(" > Restoring Model...")
            model.load_state_dict(checkpoint['model'])
            print(" > Restoring Optimizer...")
            optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                print(" > Restoring LR Scheduler...")
                scheduler.load_state_dict(checkpoint['scheduler'])
                # NOTE: Not sure if necessary
                scheduler.optimizer = optimizer
            if "scaler" in checkpoint and c.mixed_precision:
                print(" > Restoring AMP Scaler...")
                scaler.load_state_dict(checkpoint["scaler"])
        except RuntimeError:
            # retore only matching layers.
            print(" > Partial model initialization...")
            model_dict = model.state_dict()
            model_dict = set_init_dict(model_dict, checkpoint['model'], c)
            model.load_state_dict(model_dict)
            del model_dict

        # reset lr if not countinuining training.
        for group in optimizer.param_groups:
            group['lr'] = c.lr

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

    num_params = count_parameters(model)
    print(" > WaveGrad has {} parameters".format(num_params), flush=True)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    global_step = args.restore_step
    for epoch in range(0, c.epochs):
        c_logger.print_epoch_start(epoch, c.epochs)
        _, global_step = train(model, criterion, optimizer,
                               scheduler, scaler, ap, global_step,
                               epoch)
        eval_avg_loss_dict = evaluate(model, criterion, ap,
                                      global_step, epoch)
        c_logger.print_epoch_end(epoch, eval_avg_loss_dict)
        target_loss = eval_avg_loss_dict[c.target_loss]
        best_loss = save_best_model(target_loss,
                                    best_loss,
                                    model,
                                    optimizer,
                                    scheduler,
                                    None,
                                    None,
                                    None,
                                    global_step,
                                    epoch,
                                    OUT_PATH,
                                    model_losses=eval_avg_loss_dict,
                                    scaler=scaler.state_dict() if c.mixed_precision else None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help=
        'Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument('--config_path',
                        type=str,
                        help='Path to config file for training.',
                        required='--continue_path' not in sys.argv)
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
        list_of_files = glob.glob(
            args.continue_path +
            "/*.pth.tar")  # * means all if need specific format then *.csv
        latest_model_file = max(list_of_files, key=os.path.getctime)
        args.restore_path = latest_model_file
        print(f" > Training continues for {args.restore_path}")

    # setup output paths and read configs
    c = load_config(args.config_path)
    # check_config(c)
    _ = os.path.dirname(os.path.realpath(__file__))

    # DISTRIBUTED
    if c.mixed_precision:
        print("   >  Mixed precision is enabled")

    OUT_PATH = args.continue_path
    if args.continue_path == '':
        OUT_PATH = create_experiment_folder(c.output_path, c.run_name,
                                            args.debug)

    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')

    c_logger = ConsoleLogger()

    if args.rank == 0:
        os.makedirs(AUDIO_PATH, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_model_files(c,  args.config_path,
                         OUT_PATH, new_fields)
        os.chmod(AUDIO_PATH, 0o775)
        os.chmod(OUT_PATH, 0o775)

        LOG_DIR = OUT_PATH
        tb_logger = TensorboardLogger(LOG_DIR, model_name='VOCODER')

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
