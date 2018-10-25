import os
import sys
import time
import shutil
import torch
import argparse
import importlib
import traceback
import numpy as np

import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from utils.generic_utils import (
    synthesis, remove_experiment_folder, create_experiment_folder,
    save_checkpoint, save_best_model, load_config, lr_decay, count_parameters,
    check_update, get_commit_hash, sequence_mask, AnnealLR)
from utils.visual import plot_alignment, plot_spectrogram
from models.tacotron import Tacotron
from layers.losses import L1LossMasked
from utils.audio import AudioProcessor

torch.manual_seed(1)
# torch.set_num_threads(4)
use_cuda = torch.cuda.is_available()


def train(model, criterion, criterion_st, data_loader, optimizer, optimizer_st,
          scheduler, ap, epoch):
    model = model.train()
    epoch_time = 0
    avg_linear_loss = 0
    avg_mel_loss = 0
    avg_stop_loss = 0
    avg_step_time = 0
    print(" | > Epoch {}/{}".format(epoch, c.epochs), flush=True)
    n_priority_freq = int(3000 / (c.sample_rate * 0.5) * c.num_freq)
    batch_n_iter = int(len(data_loader.dataset) / c.batch_size)
    for num_iter, data in enumerate(data_loader):
        start_time = time.time()

        # setup input data
        text_input = data[0]
        text_lengths = data[1]
        linear_input = data[2]
        mel_input = data[3]
        mel_lengths = data[4]
        stop_targets = data[5]

        # set stop targets view, we predict a single stop token per r frames prediction
        stop_targets = stop_targets.view(text_input.shape[0],
                                         stop_targets.size(1) // c.r, -1)
        stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float()

        current_step = num_iter + args.restore_step + \
            epoch * len(data_loader) + 1

        # setup lr
        scheduler.step()
        optimizer.zero_grad()
        optimizer_st.zero_grad()

        # dispatch data to GPU
        if use_cuda:
            text_input = text_input.cuda()
            text_lengths = text_lengths.cuda()
            mel_input = mel_input.cuda()
            mel_lengths = mel_lengths.cuda()
            linear_input = linear_input.cuda()
            stop_targets = stop_targets.cuda()

        # compute mask for padding
        mask = sequence_mask(text_lengths)

        # forward pass
        if use_cuda:
            mel_output, linear_output, alignments, stop_tokens = torch.nn.parallel.data_parallel(
                model, (text_input, mel_input, mask))
        else:
            mel_output, linear_output, alignments, stop_tokens = model(text_input, mel_input, mask)

        # loss computation
        stop_loss = criterion_st(stop_tokens, stop_targets)
        mel_loss = criterion(mel_output, mel_input, mel_lengths)
        linear_loss = 0.5 * criterion(linear_output, linear_input, mel_lengths)\
            + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                              linear_input[:, :, :n_priority_freq],
                              mel_lengths)
        loss = mel_loss + linear_loss

        # backpass and check the grad norm for spec losses
        loss.backward(retain_graph=True)
        for group in optimizer.param_groups:
            for param in group['params']:
                param.data = param.data.add(-c.wd * group['lr'], param.data)
        grad_norm, skip_flag = check_update(model, 1)
        if skip_flag:
            optimizer.zero_grad()
            print(" | > Iteration skipped!!", flush=True)
            continue
        optimizer.step()

        # backpass and check the grad norm for stop loss
        stop_loss.backward()
        for group in optimizer_st.param_groups:
            for param in group['params']:
                param.data = param.data.add(-c.wd * group['lr'], param.data)
        grad_norm_st, skip_flag = check_update(model.decoder.stopnet, 0.5)
        if skip_flag:
            optimizer_st.zero_grad()
            print(" | | > Iteration skipped fro stopnet!!")
            continue
        optimizer_st.step()

        step_time = time.time() - start_time
        epoch_time += step_time

        if current_step % c.print_step == 0:
            print(
                " | | > Step:{}/{}  GlobalStep:{}  TotalLoss:{:.5f}  LinearLoss:{:.5f}  "
                "MelLoss:{:.5f}  StopLoss:{:.5f}  GradNorm:{:.5f}  "
                "GradNormST:{:.5f}  StepTime:{:.2f}".format(
                    num_iter, batch_n_iter, current_step, loss.item(),
                    linear_loss.item(), mel_loss.item(), stop_loss.item(),
                    grad_norm, grad_norm_st, step_time),
                flush=True)

        avg_linear_loss += linear_loss.item()
        avg_mel_loss += mel_loss.item()
        avg_stop_loss += stop_loss.item()
        avg_step_time += step_time

        # Plot Training Iter Stats
        tb.add_scalar('TrainIterLoss/TotalLoss', loss.item(), current_step)
        tb.add_scalar('TrainIterLoss/LinearLoss', linear_loss.item(),
                      current_step)
        tb.add_scalar('TrainIterLoss/MelLoss', mel_loss.item(), current_step)
        tb.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'],
                      current_step)
        tb.add_scalar('Params/GradNorm', grad_norm, current_step)
        tb.add_scalar('Params/GradNormSt', grad_norm_st, current_step)
        tb.add_scalar('Time/StepTime', step_time, current_step)

        if current_step % c.save_step == 0:
            if c.checkpoint:
                # save model
                save_checkpoint(model, optimizer, optimizer_st,
                                linear_loss.item(), OUT_PATH, current_step,
                                epoch)

            # Diagnostic visualizations
            const_spec = linear_output[0].data.cpu().numpy()
            gt_spec = linear_input[0].data.cpu().numpy()

            const_spec = plot_spectrogram(const_spec, ap)
            gt_spec = plot_spectrogram(gt_spec, ap)
            tb.add_figure('Visual/Reconstruction', const_spec, current_step)
            tb.add_figure('Visual/GroundTruth', gt_spec, current_step)

            align_img = alignments[0].data.cpu().numpy()
            align_img = plot_alignment(align_img)
            tb.add_figure('Visual/Alignment', align_img, current_step)

            # Sample audio
            audio_signal = linear_output[0].data.cpu().numpy()
            ap.griffin_lim_iters = 60
            audio_signal = ap.inv_spectrogram(audio_signal.T)
            try:
                tb.add_audio(
                    'SampleAudio',
                    audio_signal,
                    current_step,
                    sample_rate=c.sample_rate)
            except:
                pass

    avg_linear_loss /= (num_iter + 1)
    avg_mel_loss /= (num_iter + 1)
    avg_stop_loss /= (num_iter + 1)
    avg_total_loss = avg_mel_loss + avg_linear_loss + avg_stop_loss
    avg_step_time /= (num_iter + 1)

    # print epoch stats
    print(
        " | | > EPOCH END -- GlobalStep:{}  AvgTotalLoss:{:.5f}  "
        "AvgLinearLoss:{:.5f}  AvgMelLoss:{:.5f}  "
        "AvgStopLoss:{:.5f}  EpochTime:{:.2f}  "
        "AvgStepTime:{:.2f}".format(current_step, avg_total_loss,
                                    avg_linear_loss, avg_mel_loss,
                                    avg_stop_loss, epoch_time, avg_step_time),
        flush=True)

    # Plot Training Epoch Stats
    tb.add_scalar('TrainEpochLoss/TotalLoss', avg_total_loss, current_step)
    tb.add_scalar('TrainEpochLoss/LinearLoss', avg_linear_loss, current_step)
    tb.add_scalar('TrainEpochLoss/MelLoss', avg_mel_loss, current_step)
    tb.add_scalar('TrainEpochLoss/StopLoss', avg_stop_loss, current_step)
    tb.add_scalar('Time/EpochTime', epoch_time, epoch)
    epoch_time = 0
    return avg_linear_loss, current_step


def evaluate(model, criterion, criterion_st, data_loader, ap, current_step):
    model = model.eval()
    epoch_time = 0
    avg_linear_loss = 0
    avg_mel_loss = 0
    avg_stop_loss = 0
    print(" | > Validation")
    test_sentences = [
        "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
        "Be a voice, not an echo.",
        "I'm sorry Dave. I'm afraid I can't do that.",
        "This cake is great. It's so delicious and moist."
    ]
    n_priority_freq = int(3000 / (c.sample_rate * 0.5) * c.num_freq)
    with torch.no_grad():
        if data_loader is not None:
            for num_iter, data in enumerate(data_loader):
                start_time = time.time()

                # setup input data
                text_input = data[0]
                text_lengths = data[1]
                linear_input = data[2]
                mel_input = data[3]
                mel_lengths = data[4]
                stop_targets = data[5]

                # set stop targets view, we predict a single stop token per r frames prediction
                stop_targets = stop_targets.view(text_input.shape[0],
                                                 stop_targets.size(1) // c.r,
                                                 -1)
                stop_targets = (stop_targets.sum(2) > 0.0).unsqueeze(2).float()

                # dispatch data to GPU
                if use_cuda:
                    text_input = text_input.cuda()
                    mel_input = mel_input.cuda()
                    mel_lengths = mel_lengths.cuda()
                    linear_input = linear_input.cuda()
                    stop_targets = stop_targets.cuda()

                # forward pass
                mel_output, linear_output, alignments, stop_tokens =\
                    model.forward(text_input, mel_input)

                # loss computation
                stop_loss = criterion_st(stop_tokens, stop_targets)
                mel_loss = criterion(mel_output, mel_input, mel_lengths)
                linear_loss = 0.5 * criterion(linear_output, linear_input, mel_lengths) \
                    + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                    linear_input[:, :, :n_priority_freq],
                                    mel_lengths)
                loss = mel_loss + linear_loss + stop_loss

                step_time = time.time() - start_time
                epoch_time += step_time

                if num_iter % c.print_step == 0:
                    print(
                        " | | > TotalLoss: {:.5f}   LinearLoss: {:.5f}   MelLoss:{:.5f}  "
                        "StopLoss: {:.5f}  ".format(loss.item(),
                                                    linear_loss.item(),
                                                    mel_loss.item(),
                                                    stop_loss.item()),
                        flush=True)

                avg_linear_loss += linear_loss.item()
                avg_mel_loss += mel_loss.item()
                avg_stop_loss += stop_loss.item()

            # Diagnostic visualizations
            idx = np.random.randint(mel_input.shape[0])
            const_spec = linear_output[idx].data.cpu().numpy()
            gt_spec = linear_input[idx].data.cpu().numpy()
            align_img = alignments[idx].data.cpu().numpy()
            const_spec = plot_spectrogram(const_spec, ap)
            gt_spec = plot_spectrogram(gt_spec, ap)
            align_img = plot_alignment(align_img)

            tb.add_figure('ValVisual/Reconstruction', const_spec, current_step)
            tb.add_figure('ValVisual/GroundTruth', gt_spec, current_step)
            tb.add_figure('ValVisual/ValidationAlignment', align_img,
                          current_step)

            # Sample audio
            audio_signal = linear_output[idx].data.cpu().numpy()
            ap.griffin_lim_iters = 60
            audio_signal = ap.inv_spectrogram(audio_signal.T)
            try:
                tb.add_audio(
                    'ValSampleAudio',
                    audio_signal,
                    current_step,
                    sample_rate=c.sample_rate)
            except:
                # sometimes audio signal is out of boundaries
                pass

            # compute average losses
            avg_linear_loss /= (num_iter + 1)
            avg_mel_loss /= (num_iter + 1)
            avg_stop_loss /= (num_iter + 1)
            avg_total_loss = avg_mel_loss + avg_linear_loss + avg_stop_loss

            # Plot Learning Stats
            tb.add_scalar('ValEpochLoss/TotalLoss', avg_total_loss,
                          current_step)
            tb.add_scalar('ValEpochLoss/LinearLoss', avg_linear_loss,
                          current_step)
            tb.add_scalar('ValEpochLoss/MelLoss', avg_mel_loss, current_step)
            tb.add_scalar('ValEpochLoss/Stop_loss', avg_stop_loss,
                          current_step)

    # test sentences
    ap.griffin_lim_iters = 60
    for idx, test_sentence in enumerate(test_sentences):
        try:
            wav, linear_spec, alignments = synthesis(model, ap, test_sentence,
                                                     use_cuda, c.text_cleaner)

            file_path = os.path.join(AUDIO_PATH, str(current_step))
            os.makedirs(file_path, exist_ok=True)
            file_path = os.path.join(file_path,
                                     "TestSentence_{}.wav".format(idx))
            ap.save_wav(wav, file_path)

            wav_name = 'TestSentences/{}'.format(idx)
            tb.add_audio(
                wav_name, wav, current_step, sample_rate=c.sample_rate)
            align_img = alignments[0].data.cpu().numpy()
            linear_spec = plot_spectrogram(linear_spec, ap)
            align_img = plot_alignment(align_img)
            tb.add_figure('TestSentences/{}_Spectrogram'.format(idx),
                          linear_spec, current_step)
            tb.add_figure('TestSentences/{}_Alignment'.format(idx), align_img,
                          current_step)
        except:
            print(" !! Error as creating Test Sentence -", idx)
            pass
    return avg_linear_loss


def main(args):
    dataset = importlib.import_module('datasets.' + c.dataset)
    Dataset = getattr(dataset, 'MyDataset')
    audio = importlib.import_module('utils.' + c.audio_processor)
    AudioProcessor = getattr(audio, 'AudioProcessor')

    ap = AudioProcessor(
        sample_rate=c.sample_rate,
        num_mels=c.num_mels,
        min_level_db=c.min_level_db,
        frame_shift_ms=c.frame_shift_ms,
        frame_length_ms=c.frame_length_ms,
        ref_level_db=c.ref_level_db,
        num_freq=c.num_freq,
        power=c.power,
        preemphasis=c.preemphasis)

    # Setup the dataset
    train_dataset = Dataset(
        c.data_path,
        c.meta_file_train,
        c.r,
        c.text_cleaner,
        ap=ap,
        batch_group_size=8*c.batch_size,
        min_seq_len=c.min_seq_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=c.batch_size,
        shuffle=False,
        collate_fn=train_dataset.collate_fn,
        drop_last=False,
        num_workers=c.num_loader_workers,
        pin_memory=True)

    if c.run_eval:
        val_dataset = Dataset(
            c.data_path, c.meta_file_val, c.r, c.text_cleaner, ap=ap, batch_group_size=0)

        val_loader = DataLoader(
            val_dataset,
            batch_size=c.eval_batch_size,
            shuffle=False,
            collate_fn=val_dataset.collate_fn,
            drop_last=False,
            num_workers=4,
            pin_memory=True)
    else:
        val_loader = None

    model = Tacotron(c.embedding_size, ap.num_freq, c.num_mels, c.r)
    print(" | > Num output units : {}".format(ap.num_freq), flush=True)

    optimizer = optim.Adam(model.parameters(), lr=c.lr, weight_decay=0)
    optimizer_st = optim.Adam(
        model.decoder.stopnet.parameters(), lr=c.lr, weight_decay=0)

    criterion = L1LossMasked()
    criterion_st = nn.BCELoss()

    if args.restore_path:
        checkpoint = torch.load(args.restore_path)
        model.load_state_dict(checkpoint['model'])
        if use_cuda:
            model = model.cuda()
            criterion.cuda()
            criterion_st.cuda()
        optimizer.load_state_dict(checkpoint['optimizer'])
        # optimizer_st.load_state_dict(checkpoint['optimizer_st'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
        print(
            " > Model restored from step %d" % checkpoint['step'], flush=True)
        start_epoch = checkpoint['step'] // len(train_loader)
        best_loss = checkpoint['linear_loss']
        args.restore_step = checkpoint['step']
    else:
        args.restore_step = 0
        print("\n > Starting a new training", flush=True)
        if use_cuda:
            model = model.cuda()
            criterion.cuda()
            criterion_st.cuda()

    scheduler = AnnealLR(optimizer, warmup_steps=c.warmup_steps, last_epoch= (args.restore_step-1))
    num_params = count_parameters(model)
    print(" | > Model has {} parameters".format(num_params), flush=True)

    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

    if 'best_loss' not in locals():
        best_loss = float('inf')

    for epoch in range(0, c.epochs):
        train_loss, current_step = train(model, criterion, criterion_st,
                                         train_loader, optimizer, optimizer_st,
                                         scheduler, ap, epoch)
        val_loss = evaluate(model, criterion, criterion_st, val_loader, ap,
                            current_step)
        print(
            " | > Train Loss: {:.5f}   Validation Loss: {:.5f}".format(
                train_loss, val_loss),
            flush=True)
        best_loss = save_best_model(model, optimizer, train_loss, best_loss,
                                    OUT_PATH, current_step, epoch)
         # shuffle batch groups
        train_loader.dataset.sort_frames()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Folder path to checkpoints',
        default=0)
    parser.add_argument(
        '--config_path',
        type=str,
        help='path to config file for training',
    )
    parser.add_argument(
        '--debug',
        type=bool,
        default=False,
        help='do not ask for git has before run.')
    parser.add_argument(
        '--data_path',
        type=str,
        default='',
        help='data path to overrite config.json')
    args = parser.parse_args()

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))
    OUT_PATH = os.path.join(_, c.output_path)
    OUT_PATH = create_experiment_folder(OUT_PATH, c.model_name, args.debug)
    CHECKPOINT_PATH = os.path.join(OUT_PATH, 'checkpoints')
    AUDIO_PATH = os.path.join(OUT_PATH, 'test_audios')
    os.makedirs(AUDIO_PATH, exist_ok=True)
    shutil.copyfile(args.config_path, os.path.join(OUT_PATH, 'config.json'))

    if args.data_path != "":
        c.data_path = args.data_path

    # setup tensorboard
    LOG_DIR = OUT_PATH
    tb = SummaryWriter(LOG_DIR)

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
