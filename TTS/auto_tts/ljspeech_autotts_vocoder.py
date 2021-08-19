import argparse

from TTS.auto_tts.complete_recipes import VocoderExamples


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
    parser.add_argument("--model", type=str, help="This is the model you want to train with")
    parser.add_argument("--gen_learning_rate", type=float, default=0.0001)
    parser.add_argument("--disc_learning_rate", type=float, default=0.0001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--mixed_precision",
        dest="location_attention",
        action="store_true",
        help="This turns on location attention for tacotron2 models, recommended to turn on.",
    )

    parser.set_defaults(mixed_precision=False, forward_attention=False, location_attention=False)

    args = parser.parse_args()
    args = vars(args)
    trainer = VocoderExamples(
        args["data_path"],
        args["batch_size"],
        args["output_path"],
        args["mixed_precision"],
        args["gen_learning_rate"],
        args["disc_learning_rate"],
        args["epochs"],
    )
    model = trainer.ljpseechAutoTts(args["model"])
    model.fit()


if __name__ == "__main__":
    main()
