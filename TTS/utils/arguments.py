#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument parser for training scripts."""

import argparse
import re
import glob
import os

from TTS.utils.generic_utils import (
    create_experiment_folder, get_git_branch)
from TTS.utils.console_logger import ConsoleLogger
from TTS.utils.io import copy_model_files, load_config
from TTS.utils.tensorboard_logger import TensorboardLogger

from TTS.tts.utils.generic_utils import check_config_tts


def parse_arguments(argv):
    """Parse command line arguments of training scripts.

    Parameters
    ----------
    argv : list
        This is a list of input arguments as given by sys.argv

    Returns
    -------
    argparse.Namespace
        Parsed arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--continue_path",
        type=str,
        help=("Training output folder to continue training. Used to continue "
              "a training. If it is used, 'config_path' is ignored."),
        default="",
        required="--config_path" not in argv)
    parser.add_argument(
        "--restore_path",
        type=str,
        help="Model file to be restored. Use to finetune a model.",
        default="")
    parser.add_argument(
        "--config_path",
        type=str,
        help="Path to config file for training.",
        required="--continue_path" not in argv)
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Do not verify commit integrity to run training.")
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="DISTRIBUTED: process rank for distributed training.")
    parser.add_argument(
        "--group_id",
        type=str,
        default="",
        help="DISTRIBUTED: process group id.")

    return parser.parse_args()


def get_last_checkpoint(path):
    """Get latest checkpoint from a list of filenames.

    It is based on globbing for `*.pth.tar` and the RegEx
    `checkpoint_([0-9]+)`.

    Parameters
    ----------
    path : list
        Path to files to be compared.

    Raises
    ------
    ValueError
        If no checkpoint files are found.

    Returns
    -------
    last_checkpoint : str
        Last checkpoint filename.

    """
    last_checkpoint_num = 0
    last_checkpoint = None
    filenames = glob.glob(
        os.path.join(path, "/*.pth.tar"))
    for filename in filenames:
        try:
            checkpoint_num = int(
                re.search(r"checkpoint_([0-9]+)", filename).groups()[0])
            if checkpoint_num > last_checkpoint_num:
                last_checkpoint_num = checkpoint_num
                last_checkpoint = filename
        except AttributeError:  # if there's no match in the filename
            pass
    if last_checkpoint is None:
        raise ValueError(f"No checkpoints in {path}!")
    return last_checkpoint


def process_args(args, model_type):
    """Process parsed comand line arguments.

    Parameters
    ----------
    args : argparse.Namespace or dict like
        Parsed input arguments.
    model_type : str
        Model type used to check config parameters and setup the TensorBoard
        logger. One of:
            - tacotron
            - glow_tts
            - speedy_speech
            - gan
            - wavegrad
            - wavernn

    Raises
    ------
    ValueError
        If `model_type` is not one of implemented choices.

    Returns
    -------
    c : TTS.utils.io.AttrDict
        Config paramaters.
    out_path : str
        Path to save models and logging.
    audio_path : str
        Path to save generated test audios.
    c_logger : TTS.utils.console_logger.ConsoleLogger
        Class that does logging to the console.
    tb_logger : TTS.utils.tensorboard.TensorboardLogger
        Class that does the TensorBoard loggind.

    """
    if args.continue_path != "":
        args.output_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, "config.json")
        list_of_files = glob.glob(
            os.path.join(args.continue_path, "*.pth.tar")
            )  # * means all if need specific format then *.csv
        args.restore_path = max(list_of_files, key=os.path.getctime)
        # checkpoint number based continuing
        # args.restore_path = get_last_checkpoint(args.continue_path)
        print(f" > Training continues for {args.restore_path}")

    # setup output paths and read configs
    c = load_config(args.config_path)

    if model_type in "tacotron glow_tts speedy_speech":
        model_class = "TTS"
    elif model_type in "gan wavegrad wavernn":
        model_class = "VOCODER"
    else:
        raise ValueError("model type {model_type} not recognized!")

    if model_class == "TTS":
        check_config_tts(c)
    elif model_class == "VOCODER":
        print("Vocoder config checker not implemented, "
              "skipping ...")
    else:
        raise ValueError(f"model type {model_type} not recognized!")

    _ = os.path.dirname(os.path.realpath(__file__))

    if model_type in "tacotron wavegrad wavernn" and c.mixed_precision:
        print("   >  Mixed precision mode is ON")

    out_path = args.continue_path
    if args.continue_path == "":
        out_path = create_experiment_folder(c.output_path, c.run_name,
                                            args.debug)

    audio_path = os.path.join(out_path, "test_audios")

    c_logger = ConsoleLogger()

    if args.rank == 0:
        os.makedirs(audio_path, exist_ok=True)
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        copy_model_files(c, args.config_path,
                         out_path, new_fields)
        os.chmod(audio_path, 0o775)
        os.chmod(out_path, 0o775)

        log_path = out_path

        tb_logger = TensorboardLogger(log_path, model_name=model_class)

        # write model desc to tensorboard
        tb_logger.tb_add_text("model-description", c["run_description"], 0)

    return c, out_path, audio_path, c_logger, tb_logger
