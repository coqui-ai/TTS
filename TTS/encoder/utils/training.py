import os
from dataclasses import dataclass, field

from coqpit import Coqpit
from trainer import TrainerArgs, get_last_checkpoint
from trainer.io import copy_model_files
from trainer.logging import logger_factory
from trainer.logging.console_logger import ConsoleLogger

from TTS.config import load_config, register_config
from TTS.tts.utils.text.characters import parse_symbols
from TTS.utils.generic_utils import get_experiment_folder_path, get_git_branch


@dataclass
class TrainArgs(TrainerArgs):
    config_path: str = field(default=None, metadata={"help": "Path to the config file."})


def getarguments():
    train_config = TrainArgs()
    parser = train_config.init_argparse(arg_prefix="")
    return parser


def process_args(args, config=None):
    """Process parsed comand line arguments and initialize the config if not provided.
    Args:
        args (argparse.Namespace or dict like): Parsed input arguments.
        config (Coqpit): Model config. If none, it is generated from `args`. Defaults to None.
    Returns:
        c (TTS.utils.io.AttrDict): Config paramaters.
        out_path (str): Path to save models and logging.
        audio_path (str): Path to save generated test audios.
        c_logger (TTS.utils.console_logger.ConsoleLogger): Class that does
            logging to the console.
        dashboard_logger (WandbLogger or TensorboardLogger): Class that does the dashboard Logging
    TODO:
        - Interactive config definition.
    """
    if isinstance(args, tuple):
        args, coqpit_overrides = args
    if args.continue_path:
        # continue a previous training from its output folder
        experiment_path = args.continue_path
        args.config_path = os.path.join(args.continue_path, "config.json")
        args.restore_path, best_model = get_last_checkpoint(args.continue_path)
        if not args.best_path:
            args.best_path = best_model
    # init config if not already defined
    if config is None:
        if args.config_path:
            # init from a file
            config = load_config(args.config_path)
        else:
            # init from console args
            from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

            config_base = BaseTrainingConfig()
            config_base.parse_known_args(coqpit_overrides)
            config = register_config(config_base.model)()
    # override values from command-line args
    config.parse_known_args(coqpit_overrides, relaxed_parser=True)
    experiment_path = args.continue_path
    if not experiment_path:
        experiment_path = get_experiment_folder_path(config.output_path, config.run_name)
    audio_path = os.path.join(experiment_path, "test_audios")
    config.output_log_path = experiment_path
    # setup rank 0 process in distributed training
    dashboard_logger = None
    if args.rank == 0:
        new_fields = {}
        if args.restore_path:
            new_fields["restore_path"] = args.restore_path
        new_fields["github_branch"] = get_git_branch()
        # if model characters are not set in the config file
        # save the default set to the config file for future
        # compatibility.
        if config.has("characters") and config.characters is None:
            used_characters = parse_symbols()
            new_fields["characters"] = used_characters
        copy_model_files(config, experiment_path, new_fields)
        dashboard_logger = logger_factory(config, experiment_path)
    c_logger = ConsoleLogger()
    return config, experiment_path, audio_path, c_logger, dashboard_logger


def init_arguments():
    train_config = TrainArgs()
    parser = train_config.init_argparse(arg_prefix="")
    return parser


def init_training(config: Coqpit = None):
    """Initialization of a training run."""
    parser = init_arguments()
    args = parser.parse_known_args()
    config, OUT_PATH, AUDIO_PATH, c_logger, dashboard_logger = process_args(args, config)
    return args[0], config, OUT_PATH, AUDIO_PATH, c_logger, dashboard_logger
