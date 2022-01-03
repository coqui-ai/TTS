import os
import torch

from TTS.config import check_config_and_model_args, get_from_config_or_model_args, load_config, register_config
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor


def main():
    """Run `tts` model training directly by a `config.json` file."""
    # init trainer args
    train_args = TrainingArgs()
    parser = train_args.init_argparse(arg_prefix="")

    # override trainer args from comman-line args
    args, config_overrides = parser.parse_known_args()
    train_args.parse_args(args)

    # load config.json and register
    if args.config_path or args.continue_path:
        if args.config_path:
            # init from a file
            config = load_config(args.config_path)
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        elif args.continue_path:
            # continue from a prev experiment
            config = load_config(os.path.join(args.continue_path, "config.json"))
            if len(config_overrides) > 0:
                config.parse_known_args(config_overrides, relaxed_parser=True)
        else:
            # init from console args
            from TTS.config.shared_configs import BaseTrainingConfig  # pylint: disable=import-outside-toplevel

            config_base = BaseTrainingConfig()
            config_base.parse_known_args(config_overrides)
            config = register_config(config_base.model)()

    # load training samples
    train_samples, eval_samples = load_tts_samples(config.datasets, eval_split=True)

    # setup audio processor
    ap = AudioProcessor(**config.audio)

    # init speaker manager
    if check_config_and_model_args(config, "use_speaker_embedding", True):
        speaker_manager = SpeakerManager(data_items=train_samples + eval_samples)
        if hasattr(config, "model_args"):
            config.model_args.num_speakers = speaker_manager.num_speakers
        else:
            config.num_speakers = speaker_manager.num_speakers
    elif check_config_and_model_args(config, "use_d_vector_file", True):
        if check_config_and_model_args(config, "use_speaker_encoder_as_loss", True):
            speaker_manager = SpeakerManager(
                d_vectors_file_path=config.model_args.d_vector_file,
                encoder_model_path=config.model_args.speaker_encoder_model_path,
                encoder_config_path=config.model_args.speaker_encoder_config_path,
                use_cuda=torch.cuda.is_available(),
            )
        else:
            speaker_manager = SpeakerManager(d_vectors_file_path=get_from_config_or_model_args(config, "d_vector_file"))
        config.num_speakers = speaker_manager.num_speakers
        if hasattr(config, "model_args"):
            config.model_args.num_speakers = speaker_manager.num_speakers
    else:
        speaker_manager = None

    if check_config_and_model_args(config, "use_language_embedding", True):
        language_manager = LanguageManager(config=config)
        if hasattr(config, "model_args"):
            config.model_args.num_languages = language_manager.num_languages
        else:
            config.num_languages = language_manager.num_languages
    else:
        language_manager = None

    # init the model from config
    model = setup_model(config, speaker_manager, language_manager)

    # init the trainer and 🚀
    trainer = Trainer(
        train_args,
        config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        training_assets={"audio_processor": ap},
        parse_command_line_args=False,
    )
    trainer.fit()


if __name__ == "__main__":
    main()
