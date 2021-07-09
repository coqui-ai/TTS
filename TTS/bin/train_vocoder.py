import os
import sys
import traceback

from TTS.trainer import Trainer, init_training
from TTS.utils.generic_utils import remove_experiment_folder


def main():
    try:
        args, config, output_path, _, c_logger, dashboard_logger = init_training(sys.argv)
        trainer = Trainer(args, config, output_path, c_logger, dashboard_logger)
        trainer.fit()
    except KeyboardInterrupt:
        remove_experiment_folder(output_path)
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)  # pylint: disable=protected-access
    except Exception:  # pylint: disable=broad-except
        remove_experiment_folder(output_path)
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
