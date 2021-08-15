import sys

from TTS.trainer import Trainer, init_training


def main():
    """Run 🐸TTS trainer from terminal. This is also necessary to run DDP training by ```distribute.py```"""
    args, config, output_path, _, c_logger, dashboard_logger = init_training(sys.argv)
    trainer = Trainer(args, config, output_path, c_logger, dashboard_logger, cudnn_benchmark=False)
    trainer.fit()


if __name__ == "__main__":
    main()
