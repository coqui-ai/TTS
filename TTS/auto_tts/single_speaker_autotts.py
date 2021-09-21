import argparse

from TTS.auto_tts.complete_recipes import TtsAutoTrainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to the dataset")
    parser.add_argument("--dataset", type=str, required=True, help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument(
        "--output_path", type=str, default="./", help="path you want to store your model and config in."
    )
    parser.add_argument(
        "--mixed_precision", dest="mixed_precision", action="store_true", help="This turns on mixed precision training."
    )
    parser.add_argument("--model", type=str, required=True, help="This is the model you want to train with, c")
    parser.add_argument("--learning_rate", type=float, default=0.001)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument(
        "--tacotron2_model_type",
        type=str,
        help="this is the type of tactron model you want to train. options are 'double decoder consistency' and 'dynamic convolution attention'. This is only for tacotron2 models",
    )
    parser.add_argument(
        "--glow_tts_encoder",
        type=str,
        default=None,
        help="the type of encoder glow tts will train with, defaults to transformer encoder.",
    )
    parser.add_argument("--stats_path", type=str, default=None, help="stats path for audio config.")
    parser.add_argument(
        "--forward_attention",
        dest="forward_attention",
        action="store_true",
        help="This turns on foward attention for tacotron2 models.",
    )
    parser.add_argument(
        "--location_attention",
        dest="location_attention",
        action="store_true",
        help="This turns on location attention for tacotron2 models, recommended to turn on.",
    )

    parser.set_defaults(mixed_precision=False, forward_attention=False, location_attention=False)

    args = parser.parse_args()
    args = vars(args)
    trainer = TtsAutoTrainer(
        args["data_path"],
        args["dataset"],
        args["batch_size"],
        args["output_path"],
        args["mixed_precision"],
        args["learning_rate"],
        args["epochs"],
    )
    model = trainer.single_speaker_autotts(
        args["model"],
        args["stats_path"],
        args["tacotron2_model_type"],
        args["glow_tts_encoder"],
        args["forward_attention"],
        args["location_attention"],
    )
    model.fit()


if __name__ == "__main__":
    main()
