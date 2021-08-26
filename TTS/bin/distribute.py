#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import pathlib
import subprocess
import time

import torch

from TTS.trainer import TrainingArgs


def main():
    """
    Call train.py as a new process and pass command arguments
    """
    parser = TrainingArgs().init_argparse(arg_prefix="")
    parser.add_argument("--script", type=str, help="Target training script to distibute.")
    args, unargs = parser.parse_known_args()

    num_gpus = torch.cuda.device_count()
    group_id = time.strftime("%Y_%m_%d-%H%M%S")

    # set arguments for train.py
    folder_path = pathlib.Path(__file__).parent.absolute()
    if os.path.exists(os.path.join(folder_path, args.script)):
        command = [os.path.join(folder_path, args.script)]
    else:
        command = [args.script]
    command.append("--continue_path={}".format(args.continue_path))
    command.append("--restore_path={}".format(args.restore_path))
    command.append("--config_path={}".format(args.config_path))
    command.append("--group_id=group_{}".format(group_id))
    command.append("--use_ddp=true")
    command += unargs
    command.append("")

    # run processes
    processes = []
    for i in range(num_gpus):
        my_env = os.environ.copy()
        my_env["PYTHON_EGG_CACHE"] = "/tmp/tmp{}".format(i)
        command[-1] = "--rank={}".format(i)
        # prevent stdout for processes with rank != 0
        stdout = None
        p = subprocess.Popen(["python3"] + command, stdout=stdout, env=my_env)  # pylint: disable=consider-using-with
        processes.append(p)
        print(command)

    for p in processes:
        p.wait()


if __name__ == "__main__":
    main()
