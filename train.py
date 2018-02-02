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
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter

from utils.generic_utils import (Progbar, remove_experiment_folder,
                                 create_experiment_folder, save_checkpoint,
                                 load_config, lr_decay)
from utils.model import get_param_size
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

    model = Tacotron(c.embedding_size,
                     c.hidden_size,
                     c.num_mels,
                     c.num_freq,
                     c.r)
    if use_cuda:
        model = nn.DataParallel(model.cuda())

    optimizer = optim.Adam(model.parameters(), lr=c.lr)

    try:
        checkpoint = torch.load(os.path.join(
            CHECKPOINT_PATH, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n > Model restored from step %d\n" % args.restore_step)

    except:
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
    for epoch in range(c.epochs):

        dataloader = DataLoader(dataset, batch_size=c.batch_size,
                                shuffle=True, collate_fn=dataset.collate_fn,
                                drop_last=True, num_workers=c.num_loader_workers)
        print("\n | > Epoch {}/{}".format(epoch, c.epochs))
        progbar = Progbar(len(dataset) / c.batch_size)

        for i, data in enumerate(dataloader):
            start_time = time.time()

            text_input = data[0]
            magnitude_input = data[1]
            mel_input = data[2]

            current_step = i + args.restore_step + epoch * len(dataloader) + 1

            # setup lr
            current_lr = lr_decay(c.lr, current_step)
            for params_group in optimizer.param_groups:
                params_group['lr'] = current_lr

            optimizer.zero_grad()

            #try:
            #    mel_input = np.concatenate((np.zeros(
            #        [c.batch_size, 1, c.num_mels], dtype=np.float32),
            #        mel_input[:, 1:, :]), axis=1)
            #except:
            #    raise TypeError("not same dimension")

            if use_cuda:
                text_input_var = Variable(torch.from_numpy(text_input).type(
                    torch.cuda.LongTensor)).cuda()
                mel_input_var = Variable(torch.from_numpy(mel_input).type(
                    torch.cuda.FloatTensor)).cuda()
                mel_spec_var = Variable(torch.from_numpy(mel_input).type(
                    torch.cuda.FloatTensor)).cuda()
                linear_spec_var = Variable(torch.from_numpy(magnitude_input)
                    .type(torch.cuda.FloatTensor)).cuda()

            else:
                text_input_var = Variable(torch.from_numpy(text_input).type(
                    torch.LongTensor),)
                mel_input_var = Variable(torch.from_numpy(mel_input).type(
                    torch.FloatTensor))
                mel_spec_var = Variable(torch.from_numpy(
                    mel_input).type(torch.FloatTensor))
                linear_spec_var = Variable(torch.from_numpy(
                    magnitude_input).type(torch.FloatTensor))

            mel_output, linear_output, alignments =\
                model.forward(text_input_var, mel_input_var)

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
            grad_norm = nn.utils.clip_grad_norm(model.parameters(), 1.)
            optimizer.step()

            step_time = time.time() - start_time
            epoch_time += step_time

            progbar.update(i+1, values=[('total_loss', loss.data[0]),
                                      ('linear_loss', linear_loss.data[0]),
                                      ('mel_loss', mel_loss.data[0]),
                                      ('grad_norm', grad_norm)])

            tb.add_scalar('Loss/TotalLoss', loss.data[0], current_step)
            tb.add_scalar('Loss/LinearLoss', linear_loss.data[0],
                          current_step)
            tb.add_scalar('Loss/MelLoss', mel_loss.data[0], current_step)

            tb.add_scalar('Params/LearningRate', optimizer.param_groups[0]['lr'],
                          current_step)
            tb.add_scalar('Params/GradNorm', grad_norm, current_step)
            tb.add_scalar('Time/StepTime', step_time, current_step)


            if current_step % c.save_step == 0:
                checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
                checkpoint_path = os.path.join(OUT_PATH, checkpoint_path)
                save_checkpoint({'model': model.state_dict(),
                                 'optimizer': optimizer.state_dict(),
                                 'step': current_step,
                                 'total_loss': loss.data[0],
                                 'linear_loss': linear_loss.data[0],
                                 'mel_loss': mel_loss.data[0],
                                 'date': datetime.date.today().strftime("%B %d, %Y")},
                                checkpoint_path)
                print("\n | > Checkpoint is saved : {}".format(checkpoint_path))

                # Diagnostic visualizations
                const_spec = linear_output[0].data.cpu()[None, :]
                gt_spec = linear_spec_var[0].data.cpu()[None, :]
                align_img = alignments[0].data.cpu().t()[None, :]
                tb.add_image('Spec/Reconstruction', const_spec, current_step)
                tb.add_image('Spec/GroundTruth', gt_spec, current_step)
                tb.add_image('Attn/Alignment', align_img, current_step)


        #lr_scheduler.step(loss.data[0])
        tb.add_scalar('Time/EpochTime', epoch_time, epoch)
        epoch_time = 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=128)
    parser.add_argument('--config_path', type=str,
                       help='path to config file for training',)
    args = parser.parse_args()
    main(args)
