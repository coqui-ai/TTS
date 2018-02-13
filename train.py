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
import numpy as np

import torch.nn as nn
from torch import optim
from torch import onnx
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from utils.generic_utils import (Progbar, remove_experiment_folder,
                                 create_experiment_folder, save_checkpoint,
                                 save_best_model, load_config, lr_decay)
from utils.model import get_param_size
from utils.visual import plot_alignment, plot_spectrogram
from datasets.LJSpeech import LJSpeechDataset
from models.tacotron import Tacotron

use_cuda = torch.cuda.is_available()

def main(args):

    # setup output paths and read configs
    c = load_config(args.config_path)
    _ = os.path.dirname(os.path.realpath(__file__))
    OUT_PATH = os.path.join(_, c.output_path)
    OUT_PATH = create_experiment_folder(OUT_PATH)
    CHECKPOINT_PATH = os.path.join(OUT_PATH, 'checkpoints')
    shutil.copyfile(args.config_path, os.path.join(OUT_PATH, 'config.json'))

    # save config to tmp place to be loaded by subsequent modules.
    file_name = str(os.getpid())
    tmp_path = os.path.join("/tmp/", file_name+'_tts')
    pickle.dump(c, open(tmp_path, "wb"))

    # setup tensorboard
    LOG_DIR = OUT_PATH
    tb = SummaryWriter(LOG_DIR)

    # Ctrl+C handler to remove empty experiment folder
    def signal_handler(signal, frame):
        print(" !! Pressed Ctrl+C !!")
        remove_experiment_folder(OUT_PATH)
        sys.exit(1)
    signal.signal(signal.SIGINT, signal_handler)

    # Setup the dataset
    dataset = LJSpeechDataset(os.path.join(c.data_path, 'metadata.csv'),
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

    dataloader = DataLoader(dataset, batch_size=c.batch_size,
                            shuffle=True, collate_fn=dataset.collate_fn,
                            drop_last=True, num_workers=c.num_loader_workers)

    # setup the model
    model = Tacotron(c.embedding_size,
                     c.hidden_size,
                     c.num_mels,
                     c.num_freq,
                     c.r)

    # plot model on tensorboard
    dummy_input = dataset.get_dummy_data()

    ## TODO: onnx does not support RNN fully yet
    # model_proto_path = os.path.join(OUT_PATH, "model.proto")
    # onnx.export(model, dummy_input, model_proto_path, verbose=True)
    # tb.add_graph_onnx(model_proto_path)

    if use_cuda:
        model = nn.DataParallel(model.cuda())

    optimizer = optim.Adam(model.parameters(), lr=c.lr)

    if args.restore_step:
        checkpoint = torch.load(os.path.join(
            args.restore_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n > Model restored from step %d\n" % args.restore_step)
        start_epoch = checkpoint['step'] // len(dataloader)
        best_loss = checkpoint['linear_loss']
    else:
        start_epoch = 0
        print("\n > Starting a new training")

    model = model.train()

    if not os.path.exists(CHECKPOINT_PATH):
        os.mkdir(CHECKPOINT_PATH)

    if use_cuda:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = nn.L1Loss()

    n_priority_freq = int(3000 / (c.sample_rate * 0.5) * c.num_freq)

    #lr_scheduler = ReduceLROnPlateau(optimizer, factor=c.lr_decay,
    #                               patience=c.lr_patience, verbose=True)
    epoch_time = 0
    best_loss = float('inf')
    for epoch in range(0, c.epochs):

        print("\n | > Epoch {}/{}".format(epoch, c.epochs))
        progbar = Progbar(len(dataset) / c.batch_size)

        for num_iter, data in enumerate(dataloader):
            start_time = time.time()

            text_input = data[0]
            text_lengths = data[1]
            linear_input = data[2]
            mel_input = data[3]

            current_step = num_iter + args.restore_step + epoch * len(dataloader) + 1

            # setup lr
            current_lr = lr_decay(c.lr, current_step)
            for params_group in optimizer.param_groups:
                params_group['lr'] = current_lr

            optimizer.zero_grad()

            # Add a single frame of zeros to Mel Specs for better end detection
            #try:
            #    mel_input = np.concatenate((np.zeros(
            #        [c.batch_size, 1, c.num_mels], dtype=np.float32),
            #        mel_input[:, 1:, :]), axis=1)
            #except:
            #    raise TypeError("not same dimension")

            # convert inputs to variables
            text_input_var = Variable(text_input)
            mel_spec_var = Variable(mel_input)
            linear_spec_var = Variable(linear_input, volatile=True)

            # sort sequence by length.
            # TODO: might be unnecessary
            sorted_lengths, indices = torch.sort(
                     text_lengths.view(-1), dim=0, descending=True)
            sorted_lengths = sorted_lengths.long().numpy()

            text_input_var = text_input_var[indices]
            mel_spec_var = mel_spec_var[indices]
            linear_spec_var = linear_spec_var[indices]

            if use_cuda:
                text_input_var = text_input_var.cuda()
                mel_spec_var = mel_spec_var.cuda()
                linear_spec_var = linear_spec_var.cuda()

            mel_output, linear_output, alignments =\
                model.forward(text_input_var, mel_spec_var,
                              input_lengths= torch.autograd.Variable(torch.cuda.LongTensor(sorted_lengths)))

            mel_loss = criterion(mel_output, mel_spec_var)
            #linear_loss = torch.abs(linear_output - linear_spec_var)
            #linear_loss = 0.5 * \
                #torch.mean(linear_loss) + 0.5 * \
                #torch.mean(linear_loss[:, :n_priority_freq, :])
            linear_loss = 0.5 * criterion(linear_output, linear_spec_var) \
                    + 0.5 * criterion(linear_output[:, :, :n_priority_freq],
                                      linear_spec_var[: ,: ,:n_priority_freq])
            loss = mel_loss + linear_loss
            # loss = loss.cuda()

            loss.backward()
            grad_norm = nn.utils.clip_grad_norm(model.parameters(), 1.)  ## TODO: maybe no need
            optimizer.step()

            step_time = time.time() - start_time
            epoch_time += step_time

            progbar.update(num_iter+1, values=[('total_loss', loss.data[0]),
                                       ('linear_loss', linear_loss.data[0]),
                                       ('mel_loss', mel_loss.data[0]),
                                       ('grad_norm', grad_norm)])

            # Plot Learning Stats
            tb.add_scalar('Loss/TotalLoss', loss.data[0], current_step)
            tb.add_scalar('Loss/LinearLoss', linear_loss.data[0],
                          current_step)
            tb.add_scalar('Loss/MelLoss', mel_loss.data[0], current_step)
            tb.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'],
                          current_step)
            tb.add_scalar('Params/GradNorm', grad_norm, current_step)
            tb.add_scalar('Time/StepTime', step_time, current_step)

            align_img = alignments[0].data.cpu().numpy()
            align_img = plot_alignment(align_img)
            tb.add_image('Attn/Alignment', align_img, current_step)

            if current_step % c.save_step == 0:

                if c.checkpoint:
                    # save model
                    save_checkpoint(model, optimizer, linear_loss.data[0],
                                    best_loss, OUT_PATH,
                                    current_step, epoch)

                # Diagnostic visualizations
                const_spec = linear_output[0].data.cpu().numpy()
                gt_spec = linear_spec_var[0].data.cpu().numpy()

                const_spec = plot_spectrogram(const_spec, dataset.ap)
                gt_spec = plot_spectrogram(gt_spec, dataset.ap)
                tb.add_image('Spec/Reconstruction', const_spec, current_step)
                tb.add_image('Spec/GroundTruth', gt_spec, current_step)

                align_img = alignments[0].data.cpu().numpy()
                align_img = plot_alignment(align_img)
                tb.add_image('Attn/Alignment', align_img, current_step)

                # Sample audio
                audio_signal = linear_output[0].data.cpu().numpy()
                dataset.ap.griffin_lim_iters = 60
                audio_signal = dataset.ap.inv_spectrogram(audio_signal.T)
                try:
                    tb.add_audio('SampleAudio', audio_signal, current_step,
                                 sample_rate=c.sample_rate)
                except:
                    print("\n > Error at audio signal on TB!!")
                    print(audio_signal.max())
                    print(audio_signal.min())


        # average loss after the epoch
        avg_epoch_loss = np.mean(
            progbar.sum_values['linear_loss'][0] / max(1, progbar.sum_values['linear_loss'][1]))
        best_loss = save_best_model(model, optimizer, avg_epoch_loss,
                                    best_loss, OUT_PATH,
                                    current_step, epoch)

        #lr_scheduler.step(loss.data[0])
        tb.add_scalar('Time/EpochTime', epoch_time, epoch)
        epoch_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=0)
    parser.add_argument('--restore_path', type=str,
                        help='Folder path to checkpoints', default=0)
    parser.add_argument('--config_path', type=str,
                       help='path to config file for training',)
    args = parser.parse_args()
    main(args)
