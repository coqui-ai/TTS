import os
import sys
import time
import datetime
import shutil
import torch
import signal
import argparse
import importlib
import pickle
import traceback
import numpy as np

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.generic_utils import (Progbar, remove_experiment_folder,
                                 create_experiment_folder, save_checkpoint,
                                 save_best_model, load_config, lr_decay,
                                 count_parameters, check_update, get_commit_hash,
                                 create_attn_mask, mk_decay)
from utils.model import get_param_size
from utils.visual import plot_alignment, plot_spectrogram
from models.tacotron import Tacotron
from layers.losses import L1LossMasked

torch.manual_seed(1)

use_cuda = torch.cuda.is_available()

parser = argparse.ArgumentParser()
parser.add_argument('--restore_path', type=str,
                    help='Folder path to checkpoints', default=0)
parser.add_argument('--config_path', type=str,
                    help='path to config file for training',)
args = parser.parse_args()

# setup output paths and read configs
c = load_config(args.config_path)
_ = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = os.path.join(_, c.output_path)
OUT_PATH = create_experiment_folder(OUT_PATH, c.model_name)
CHECKPOINT_PATH = os.path.join(OUT_PATH, 'checkpoints')
shutil.copyfile(args.config_path, os.path.join(OUT_PATH, 'config.json'))

parser.add_argument('--finetine_path', type=str)
# save config to tmp place to be loaded by subsequent modules.
file_name = str(os.getpid())
tmp_path = os.path.join("/tmp/", file_name+'_tts')
pickle.dump(c, open(tmp_path, "wb"))

# setup tensorboard
LOG_DIR = OUT_PATH
tb = SummaryWriter(LOG_DIR)

if c.priority_freq:
    n_priority_freq = int(3000 / (c.sample_rate * 0.5) * c.num_freq)
    print(" > Using num priority freq. : {}".format(n_priority_freq))
else:
    print(" > Priority freq. is disabled.")


def train(model, criterion, criterion_st, data_loader, optimizer, epoch):
    model = model.train()
    epoch_time = 0
    avg_linear_loss = 0
    avg_mel_loss = 0
    avg_stop_loss = 0
    avg_attn_loss = 0

    print(" | > Epoch {}/{}".format(epoch, c.epochs))
    progbar = Progbar(len(data_loader.dataset) / c.batch_size)
    progbar_display = {}
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        linear_spec = data[2]
        mel_spec = data[3]
        mel_lengths = data[4]
        stop_target = data[5]

        current_step = num_iter + args.restore_step + \
            epoch * len(data_loader) + 1

        # setup lr
        current_lr = lr_decay(c.lr, current_step, c.warmup_steps)
        for params_group in optimizer.param_groups:
            params_group['lr'] = current_lr

        optimizer.zero_grad()
        
        stop_target = stop_target.view(text_input.shape[0], stop_target.size(1) // c.r, -1)
        stop_target = (stop_target.sum(2) > 0.0).float()

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda()
            mel_spec = mel_spec.cuda()
            mel_lengths = mel_lengths.cuda()
            linear_spec = linear_spec.cuda()
            stop_target = stop_target.cuda()
            
        # create attention mask
        if c.mk > 0.0:
            N = text_input.shape[1]
            T = mel_spec.shape[1] // c.r
            M = create_attn_mask(N, T, 0.03)
            mk = mk_decay(c.mk, c.epochs, epoch)
        
        # forward pass
        mel_output, linear_output, alignments, stop_tokens =\
            model.forward(text_input, mel_spec)

        # loss computation
        mel_loss = criterion(mel_output, mel_spec, mel_lengths)
        linear_loss = criterion(linear_output, linear_spec, mel_lengths)
        stop_loss = criterion_st(stop_tokens, stop_target)
        if c.priority_freq:
            linear_loss =  0.5 * linear_loss\
                + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                  linear_spec[:, :, :n_priority_freq],
                                  mel_lengths)
        loss = mel_loss + linear_loss + stop_loss
        if c.mk > 0.0:
            attention_loss = criterion(alignments, M, mel_lengths)
            loss += mk * attention_loss
            avg_attn_loss += attention_loss.item()
            progbar_display['attn_loss'] = attention_loss.item()

        # backpass and check the grad norm
        loss.backward()
        grad_norm, skip_flag = check_update(model, 0.5, 100)
        if skip_flag:
            optimizer.zero_grad()
            print(" | > Iteration skipped!!")
            continue
        optimizer.step()

        step_time = time.time() - start_time
        epoch_time += step_time
        
        progbar_display['total_loss'] =  loss.item()
        progbar_display['linear_loss'] = linear_loss.item()
        progbar_display['mel_loss'] = mel_loss.item()
        progbar_display['stop_loss'] = stop_loss.item()
        progbar_display['grad_norm'] = grad_norm.item()

        # update
        progbar.update(num_iter+1, values=list(progbar_display.items()))
        avg_linear_loss += linear_loss.item()
        avg_mel_loss += mel_loss.item()
        avg_stop_loss += stop_loss.item()

        # Plot Training Iter Stats
        tb.add_scalar('TrainIterLoss/TotalLoss', loss.item(), current_step)
        tb.add_scalar('TrainIterLoss/LinearLoss', linear_loss.item(),
                      current_step)
        tb.add_scalar('TrainIterLoss/MelLoss', mel_loss.item(), current_step)
        tb.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'],
                      current_step)
        tb.add_scalar('Params/GradNorm', grad_norm, current_step)
        tb.add_scalar('Time/StepTime', step_time, current_step)

        if current_step % c.save_step == 0:
            if c.checkpoint:
                # save model
                save_checkpoint(model, optimizer, linear_loss.item(),
                                OUT_PATH, current_step, epoch)

            # Diagnostic visualizations
            const_spec = linear_output[0].data.cpu().numpy()
            gt_spec = linear_spec[0].data.cpu().numpy()

            const_spec = plot_spectrogram(const_spec, data_loader.dataset.ap)
            gt_spec = plot_spectrogram(gt_spec, data_loader.dataset.ap)
            tb.add_image('Visual/Reconstruction', const_spec, current_step)
            tb.add_image('Visual/GroundTruth', gt_spec, current_step)

            align_img = alignments[0].data.cpu().numpy()
            align_img = plot_alignment(align_img)
            tb.add_image('Visual/Alignment', align_img, current_step)

            # Sample audio
            audio_signal = linear_output[0].data.cpu().numpy()
            data_loader.dataset.ap.griffin_lim_iters = 60
            audio_signal = data_loader.dataset.ap.inv_spectrogram(
                audio_signal.T)
            try:
                tb.add_audio('SampleAudio', audio_signal, current_step,
                             sample_rate=c.sample_rate)
            except:
                # print("\n > Error at audio signal on TB!!")
                # print(audio_signal.max())
                # print(audio_signal.min())
                pass

    avg_linear_loss /= (num_iter + 1)
    avg_mel_loss /= (num_iter + 1)
    avg_stop_loss /= (num_iter + 1)
    avg_total_loss = avg_mel_loss + avg_linear_loss + avg_stop_loss

    # Plot Training Epoch Stats
    tb.add_scalar('TrainEpochLoss/TotalLoss', avg_total_loss, current_step)
    tb.add_scalar('TrainEpochLoss/LinearLoss', avg_linear_loss, current_step)
    tb.add_scalar('TrainEpochLoss/StopLoss', avg_stop_loss, current_step)
    tb.add_scalar('TrainEpochLoss/MelLoss', avg_mel_loss, current_step)
    if c.mk > 0:
        avg_attn_loss /= (num_iter + 1)
        tb.add_scalar('TrainEpochLoss/AttnLoss', avg_attn_loss, current_step)
    tb.add_scalar('Time/EpochTime', epoch_time, epoch)
    epoch_time = 0

    return avg_linear_loss, current_step


def evaluate(model, criterion, criterion_st, data_loader, current_step):
    model = model.eval()
    epoch_time = 0
    avg_linear_loss = 0
    avg_mel_loss = 0
    avg_stop_loss = 0

    print("\n | > Validation")
    progbar = Progbar(len(data_loader.dataset) / c.batch_size)
    with torch.no_grad():
        for num_iter, data in enumerate(data_loader):
            start_time = time.time()

            # setup input data
            text_input = data[0]
            text_lengths = data[1]
            linear_spec = data[2]
            mel_spec = data[3]
            mel_lengths = data[4]
            stop_target = data[5]

            stop_target = stop_target.view(text_input.shape[0], stop_target.size(1) // c.r, -1)
            stop_target = (stop_target.sum(2) > 0.0).float()
        
            # dispatch data to GPU
            if use_cuda:
                text_input = text_input.cuda()
                mel_spec = mel_spec.cuda()
                mel_lengths = mel_lengths.cuda()
                linear_spec = linear_spec.cuda()
                stop_target = stop_target.cuda()

            # forward pass
            mel_output, linear_output, alignments, stop_tokens =\
                model.forward(text_input, mel_spec)

            # loss computation
            mel_loss = criterion(mel_output, mel_spec, mel_lengths)
            linear_loss = criterion(linear_output, linear_spec, mel_lengths)
            stop_loss = criterion_st(stop_tokens, stop_target)
            if c.priority_freq:
                linear_loss =  0.5 * linear_loss\
                    + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                      linear_spec[:, :, :n_priority_freq],
                                      mel_lengths)
            loss = mel_loss + linear_loss + stop_loss

            step_time = time.time() - start_time
            epoch_time += step_time

            # update
            progbar.update(num_iter+1, values=[('total_loss', loss.item()),
                                               ('linear_loss',
                                                linear_loss.item()),
                                               ('stop_loss', stop_loss.item()),
                                               ('mel_loss', mel_loss.item())])
            sys.stdout.flush()

            avg_linear_loss += linear_loss.item()
            avg_mel_loss += mel_loss.item()
            avg_stop_loss += stop_loss.item()

    # Diagnostic visualizations
    idx = np.random.randint(mel_spec.shape[0])
    const_spec = linear_output[idx].data.cpu().numpy()
    gt_spec = linear_spec[idx].data.cpu().numpy()
    align_img = alignments[idx].data.cpu().numpy()

    const_spec = plot_spectrogram(const_spec, data_loader.dataset.ap)
    gt_spec = plot_spectrogram(gt_spec, data_loader.dataset.ap)
    align_img = plot_alignment(align_img)

    tb.add_image('ValVisual/Reconstruction', const_spec, current_step)
    tb.add_image('ValVisual/GroundTruth', gt_spec, current_step)
    tb.add_image('ValVisual/ValidationAlignment', align_img, current_step)

    # Sample audio
    audio_signal = linear_output[idx].data.cpu().numpy()
    data_loader.dataset.ap.griffin_lim_iters = 60
    audio_signal = data_loader.dataset.ap.inv_spectrogram(audio_signal.T)
    try:
        tb.add_audio('ValSampleAudio', audio_signal, current_step,
                     sample_rate=c.sample_rate)
    except:
        # print(" | > Error at audio signal on TB!!")
        # print(audio_signal.max())
        # print(audio_signal.min())
        pass

    # compute average losses
    avg_linear_loss /= (num_iter + 1)
    avg_mel_loss /= (num_iter + 1)
    avg_stop_loss /= (num_iter + 1)
    avg_total_loss = avg_mel_loss + avg_linear_loss + stop_loss

    # Plot Learning Stats
    tb.add_scalar('ValEpochLoss/TotalLoss', avg_total_loss, current_step)
    tb.add_scalar('ValEpochLoss/LinearLoss', avg_linear_loss, current_step)
    tb.add_scalar('ValEpochLoss/MelLoss', avg_mel_loss, current_step)
    tb.add_scalar('ValEpochLoss/Stop_loss', avg_stop_loss, current_step)
    return avg_linear_loss


def main(args):
    print(" > Using dataset: {}".format(c.dataset))
    mod = importlib.import_module('datasets.{}'.format(c.dataset))
    Dataset = getattr(mod, c.dataset+"Dataset")

    # Setup the dataset
    train_dataset = Dataset(os.path.join(c.data_path, c.meta_file_train),
                            os.path.join(c.data_path, 'wavs'),
                            c.r,
                            c.sample_rate,
                            c.text_cleaner,
                            c.num_mels,
                            c.min_level_db,
                            c.frame_shift_ms,
                            c.frame_length_ms,
                            c.preemphasis,
                            c.ref_level_db,
                            c.num_freq,
                            c.power,
                            min_seq_len=c.min_seq_len
                            )

    train_loader = DataLoader(train_dataset, batch_size=c.batch_size,
                              shuffle=False, collate_fn=train_dataset.collate_fn,
                              drop_last=True, num_workers=c.num_loader_workers,
                              pin_memory=True)

    val_dataset = Dataset(os.path.join(c.data_path, c.meta_file_val),
                          os.path.join(c.data_path, 'wavs'),
                          c.r,
                          c.sample_rate,
                          c.text_cleaner,
                          c.num_mels,
                          c.min_level_db,
                          c.frame_shift_ms,
                          c.frame_length_ms,
                          c.preemphasis,
                          c.ref_level_db,
                          c.num_freq,
                          c.power
                          )

    val_loader = DataLoader(val_dataset, batch_size=c.eval_batch_size,
                            shuffle=False, collate_fn=val_dataset.collate_fn,
                            drop_last=False, num_workers=4,
                            pin_memory=True)

    model = Tacotron(c.embedding_size,
                     c.num_freq,
                     c.num_mels,
                     c.r)

    if use_cuda:
        criterion = L1LossMasked().cuda()
        criterion_st = nn.BCELoss().cuda()
    else:
        criterion = L1LossMasked()
        criterion_st = nn.BCELoss()

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        model.load_state_dict(checkpoint['model'])
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(" > Model restored from step %d" % checkpoint['step'])
        start_epoch = checkpoint['step'] // len(train_loader)
        best_loss = checkpoint['linear_loss']
        start_epoch = 0
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0
        optimizer = optim.Adam(model.parameters(), lr=c.lr)
        print(" > Starting a new training")

    if use_cuda:
        print(" > Using CUDA.")
        model = nn.DataParallel(model).cuda()

    num_params = count_parameters(model)
    print(" | > Model has {} parameters".format(num_params))

    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    for epoch in range(0, c.epochs):
        train_loss, current_step = train(
            model, criterion, criterion_st, train_loader, optimizer, epoch)
        val_loss = evaluate(model, criterion, criterion_st, val_loader, current_step)
        best_loss = save_best_model(model, optimizer, val_loss,
                                    best_loss, OUT_PATH,
                                    current_step, epoch)


if __name__ == '__main__':
    # signal.signal(signal.SIGINT, signal_handler)
    try:
        main(args)
    except KeyboardInterrupt:
        remove_experiment_folder(OUT_PATH)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
    except Exception:
        remove_experiment_folder(OUT_PATH)
        traceback.print_exc()
        sys.exit(1)
