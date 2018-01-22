import os
import sys
import time
import torch
import signal
import argparse
import numpy as np

import torch.nn as nn
from torch import optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import train_config as c
from utils.generic_utils import (Progbar, remove_experiment_folder,
                                 create_experiment_folder, save_checkpoint)
from utils.model import get_param_size
from datasets.LJSpeech import LJSpeechDataset
from models.tacotron import Tacotron

use_cuda = torch.cuda.is_available()

_ = os.path.dirname(os.path.realpath(__file__))
OUT_PATH = os.path.join(_, c.output_path)
OUT_PATH = create_experiment_folder(OUT_PATH)

def signal_handler(signal, frame):
    print(" !! Pressed Ctrl+C !!")
    remove_experiment_folder(OUT_PATH)
    sys.exit(0)


def main(args):

    dataset = LJSpeechDataset(os.path.join(c.data_path, 'metadata.csv'),
                              os.path.join(c.data_path, 'wavs'),
                              c.dec_out_per_step
                             )

    model = Tacotron(c.embedding_size,
                     c.hidden_size,
                     c.num_mels,
                     c.num_freq,
                     c.dec_out_per_step)
    if use_cuda:
        model = nn.DataParallel(model.cuda())

    optimizer = optim.Adam(model.parameters(), lr=c.lr)

    try:
        checkpoint = torch.load(os.path.join(
            c.checkpoint_path, 'checkpoint_%d.pth.tar' % args.restore_step))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        print("\n > Model restored from step %d\n" % args.restore_step)

    except:
        print("\n > Starting a new training\n")

    model = model.train()

    if not os.path.exists(c.checkpoint_path):
        os.mkdir(c.checkpoint_path)

    if use_cuda:
        criterion = nn.L1Loss().cuda()
    else:
        criterion = nn.L1Loss()

    n_priority_freq = int(3000 / (c.sample_rate * 0.5) * c.num_freq)

    for epoch in range(c.epochs):

        dataloader = DataLoader(dataset, batch_size=args.batch_size,
                                shuffle=True, collate_fn=dataset.collate_fn,
                                drop_last=True, num_workers=32)
        progbar = Progbar(len(dataset) / args.batch_size)

        for i, data in enumerate(dataloader):
            text_input = data[0]
            magnitude_input = data[1]
            mel_input = data[2]

            current_step = i + args.restore_step + epoch * len(dataloader) + 1

            optimizer.zero_grad()

            try:
                mel_input = np.concatenate((np.zeros(
                    [args.batch_size, 1, c.num_mels], dtype=np.float32),
                    mel_input[:, 1:, :]), axis=1)
            except:
                raise TypeError("not same dimension")

            if use_cuda:
                text_input_var = Variable(torch.from_numpy(text_input).type(
                    torch.cuda.LongTensor), requires_grad=False).cuda()
                mel_input_var = Variable(torch.from_numpy(mel_input).type(
                    torch.cuda.FloatTensor), requires_grad=False).cuda()
                mel_spec_var = Variable(torch.from_numpy(mel_input).type(
                    torch.cuda.FloatTensor), requires_grad=False).cuda()
                linear_spec_var = Variable(torch.from_numpy(magnitude_input)
                    .type(torch.cuda.FloatTensor), requires_grad=False).cuda()

            else:
                text_input_var = Variable(torch.from_numpy(text_input).type(
                    torch.LongTensor), requires_grad=False)
                mel_input_var = Variable(torch.from_numpy(mel_input).type(
                    torch.FloatTensor), requires_grad=False)
                mel_spec_var = Variable(torch.from_numpy(
                    mel_input).type(torch.FloatTensor), requires_grad=False)
                linear_spec_var = Variable(torch.from_numpy(
                    magnitude_input).type(torch.FloatTensor),
                                          requires_grad=False)

            mel_output, linear_output, alignments =\
                model.forward(text_input_var, mel_input_var)

            mel_loss = criterion(mel_output, mel_spec_var)
            linear_loss = torch.abs(linear_output - linear_spec_var)
            linear_loss = 0.5 * \
                torch.mean(linear_loss) + 0.5 * \
                torch.mean(linear_loss[:, :n_priority_freq, :])
            loss = mel_loss + linear_loss
            loss = loss.cuda()

            start_time = time.time()

            loss.backward()

            nn.utils.clip_grad_norm(model.parameters(), 1.)

            optimizer.step()

            time_per_step = time.time() - start_time
            progbar.update(i, values=[('total_loss', loss.data[0]),
                                      ('linear_loss', linear_loss.data[0]),
                                      ('mel_loss', mel_loss.data[0])])

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
                print(" > Checkpoint is saved : {}".format(checkpoint_path))

            if current_step in c.decay_step:
                optimizer = adjust_learning_rate(optimizer, current_step)


def adjust_learning_rate(optimizer, step):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    if step == 500000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0005

    elif step == 1000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0003

    elif step == 2000000:
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.0001

    return optimizer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_step', type=int,
                        help='Global step to restore checkpoint', default=128)
    parser.add_argument('--batch_size', type=int,
                        help='Batch size', default=128)
    parser.add_argument('--config', type=str,
                       help='path to config file for training',)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    main(args)
