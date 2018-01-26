import os
import sys
import glob
import time
import shutil
import datetime
import json
import torch
import numpy as np


class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self


def load_config(config_path):
    config = AttrDict()
    config.update(json.load(open(config_path, "r")))
    return  config


def create_experiment_folder(root_path):
    """ Create a folder with the current date and time """
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I:%M%p")
    output_folder = os.path.join(root_path, date_str)
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


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


class Progbar(object):
    """Displays a progress bar.
    # Arguments
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
            else:
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
