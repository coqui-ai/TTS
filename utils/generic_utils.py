import os
import sys
import glob
import time
import shutil
import datetime
import json
import torch
import subprocess
import numpy as np
from collections import OrderedDict


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    config.update(json.load(open(config_path, "r")))
    return config


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    try:
        subprocess.check_output(['git', 'diff-index', '--quiet', 'HEAD'])   # Verify client is clean
    except:
        raise RuntimeError(" !! Commit before training to get the commit hash.")
    commit = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).decode().strip()
    print(' > Git Hash: {}'.format(commit))
    return commit


def create_experiment_folder(root_path, model_name, debug):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I:%M%p")
    if debug:
        commit_hash = 'debug'
    else: 
        commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, date_str + '-' + model_name + '-' + commit_hash)
    os.makedirs(output_folder, exist_ok=True)
    print(" > Experiment folder: {}".format(output_folder))
    return output_folder


def remove_experiment_folder(experiment_path):
    """Check folder if there is a checkpoint, otherwise remove the folder"""

    checkpoint_files = glob.glob(experiment_path+"/*.pth.tar")
    if len(checkpoint_files) < 1:
        if os.path.exists(experiment_path):
            shutil.rmtree(experiment_path)
            print(" ! Run is removed from {}".format(experiment_path))
    else:
        print(" ! Run is kept in {}".format(experiment_path))


def copy_config_file(config_file, path):
    config_name = os.path.basename(config_file)
    out_path = os.path.join(path, config_name)
    shutil.copyfile(config_file, out_path)


def _trim_model_state_dict(state_dict):
    r"""Remove 'module.' prefix from state dictionary. It is necessary as it
    is loded for the next time by model.load_state(). Otherwise, it complains
    about the torch.DataParallel()"""

    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    return new_state_dict


def save_checkpoint(model, optimizer, model_loss, out_path,
                    current_step, epoch):
    checkpoint_path = 'checkpoint_{}.pth.tar'.format(current_step)
    checkpoint_path = os.path.join(out_path, checkpoint_path)
    print("\n | > Checkpoint saving : {}".format(checkpoint_path))

    new_state_dict = _trim_model_state_dict(model.state_dict())
    state = {'model': new_state_dict,
             'optimizer': optimizer.state_dict(),
             'step': current_step,
             'epoch': epoch,
             'linear_loss': model_loss,
             'date': datetime.date.today().strftime("%B %d, %Y")}
    torch.save(state, checkpoint_path)


def save_best_model(model, optimizer, model_loss, best_loss, out_path,
                    current_step, epoch):
    if model_loss < best_loss:
        new_state_dict = _trim_model_state_dict(model.state_dict())
        state = {'model': new_state_dict,
                 'optimizer': optimizer.state_dict(),
                 'step': current_step,
                 'epoch': epoch,
                 'linear_loss': model_loss,
                 'date': datetime.date.today().strftime("%B %d, %Y")}
        best_loss = model_loss
        bestmodel_path = 'best_model.pth.tar'
        bestmodel_path = os.path.join(out_path, bestmodel_path)
        print(" | > Best model saving with loss {0:.2f} : {1:}".format(
            model_loss, bestmodel_path))
        torch.save(state, bestmodel_path)
    return best_loss


def check_update(model, grad_clip, grad_top):
    r'''Check model gradient against unexpected jumps and failures'''
    skip_flag = False
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    if np.isinf(grad_norm):
        print(" | > Gradient is INF !!")
        skip_flag = True
    elif grad_norm > grad_top:
        print(" | > Gradient is above the top limit !!")
        skip_flag = True
    return grad_norm, skip_flag


def lr_decay(init_lr, global_step, warmup_steps):
    r'''from https://github.com/r9y9/tacotron_pytorch/blob/master/train.py'''
    warmup_steps = float(warmup_steps)
    step = global_step + 1.
    lr = init_lr * warmup_steps**0.5 * np.minimum(step * warmup_steps**-1.5,
                                                  step**-0.5)
    return lr


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class Progbar(object):
    """Displays a progress bar.
    Args:
        target: Total number of steps expected, None if unknown.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=30, verbose=1, interval=0.05):
        self.width = width
        self.target = target
        self.sum_values = {}
        self.unique_values = []
        self.start = time.time()
        self.last_update = 0
        self.interval = interval
        self.total_width = 0
        self.seen_so_far = 0
        self.verbose = verbose
        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                 'ipykernel' in sys.modules)

    def update(self, current, values=None, force=False):
        """Updates the progress bar.
        # Arguments
            current: Index of current step.
            values: List of tuples (name, value_for_last_step).
                The progress bar will display averages for these values.
            force: Whether to force visual progress update.
        """
        values = values or []
        for k, v in values:
            if k not in self.sum_values:
                self.sum_values[k] = [v * (current - self.seen_so_far),
                                      current - self.seen_so_far]
                self.unique_values.append(k)
            else:
                self.sum_values[k][0] += v * (current - self.seen_so_far)
                self.sum_values[k][1] += (current - self.seen_so_far)
        self.seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self.start)
        if self.verbose == 1:
            if (not force and (now - self.last_update) < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self.total_width
            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self.total_width = len(bar)
            sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self.start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (
                        eta // 3600, (eta % 3600) // 60, eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format

            if time_per_unit >= 1:
                info += ' %.0fs/step' % time_per_unit
            elif time_per_unit >= 1e-3:
                info += ' %.0fms/step' % (time_per_unit * 1e3)
            else:
                info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self.unique_values:
                info += ' - %s:' % k
                if isinstance(self.sum_values[k], list):
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self.sum_values[k]

            self.total_width += len(info)
            if prev_total_width > self.total_width:
                info += (' ' * (prev_total_width - self.total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self.unique_values:
                    info += ' - %s:' % k
                    avg = np.mean(
                        self.sum_values[k][0] / max(1, self.sum_values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'

                sys.stdout.write(info)
                sys.stdout.flush()

        self.last_update = now

    def add(self, n, values=None):
        self.update(self.seen_so_far + n, values)
