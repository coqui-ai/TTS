import argparse

from TTS.auto_tts.complete_recipes import TtsExamples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="path to the dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_path", type=str, default="./", help="path you want to store your model and config in."
    )
    parser.add_argument(
        "--mixed_precision", dest="mixed_precision", action="store_true", help="This turns on mixed precision training."
    )
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--model_type",
        type=str,
        default="tacotron2",
        help="this is the type of tactron mod you want to train. options are 'double decoder consistency' and 'dynamic convolution attention'.",
    )
    parser.set_defaults(mixed_precision=False)

    args = parser.parse_args()
    args = vars(args)
    trainer = TtsExamples(
        args["data_path"],
        args["batch_size"],
        args["output_path"],
        args["mixed_precision"],
        args["learning_rate"],
        args["epochs"],
    )
    model = trainer.ljspeech_tacotron2(args["model"])
    model.fit()


if __name__ == "__main__":
    main()
