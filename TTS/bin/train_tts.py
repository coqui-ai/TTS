import os
import sys
import traceback
from TTS.utils.arguments import init_training
from TTS.utils.generic_utils import remove_experiment_folder
from TTS.trainer import TrainerTTS


def main():
    # try:
    args, config, OUT_PATH, AUDIO_PATH, c_logger, tb_logger = init_training(
        sys.argv)
    trainer = TrainerTTS(args,
                         config,
                         c_logger,
                         tb_logger,
                         output_path=OUT_PATH)
    trainer.fit()
    # except KeyboardInterrupt:
    #     remove_experiment_folder(OUT_PATH)
    #     try:
    #         sys.exit(0)
    #     except SystemExit:
    #         os._exit(0)  # pylint: disable=protected-access
    # except Exception:  # pylint: disable=broad-except
    #     remove_experiment_folder(OUT_PATH)
    #     traceback.print_exc()
    #     sys.exit(1)


if __name__ == "__main__":
    main()
