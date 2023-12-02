import os

# Trainer: Where the ✨️ happens.
# TrainingArgs: Defines the set of arguments of the Trainer.
from trainer import Trainer, TrainerArgs

# GlowTTSConfig: all model related values for training, validating and testing.
from TTS.tts.configs.glow_tts_config import GlowTTSConfig

# BaseDatasetConfig: defines name, formatter and path of the dataset.
from TTS.tts.configs.shared_configs import BaseAudioConfig, BaseDatasetConfig, CharactersConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.glow_tts import GlowTTS
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

# we use the same path as this script as our training folder.
output_path = "/storage/output-glowtts/"


# DEFINE DATASET CONFIG
# Set LJSpeech as our target dataset and define its path.
# You can also use a simple Dict to define the dataset and pass it to your custom formatter.
dataset_config = BaseDatasetConfig(
    formatter="bel_tts_formatter",
    meta_file_train="ipa_final_dataset.csv",
    path=os.path.join(output_path, "/storage/filtered_dataset/"),
)

characters = CharactersConfig(
    characters_class="TTS.tts.utils.text.characters.Graphemes",
    pad="_",
    eos="~",
    bos="^",
    blank="@",
    characters="Iabdfgijklmnprstuvxzɔɛɣɨɫɱʂʐʲˈː̯͡β",
    punctuations="!,.?: -‒–—…",
)

audio_config = BaseAudioConfig(
    mel_fmin=50,
    mel_fmax=8000,
    hop_length=256,
    stats_path="/storage/TTS/scale_stats.npy",
)

# INITIALIZE THE TRAINING CONFIGURATION
# Configure the model. Every config class inherits the BaseTTSConfig.
config = GlowTTSConfig(
    batch_size=96,
    eval_batch_size=32,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    use_noise_augment=True,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    print_step=50,
    print_eval=True,
    output_path=output_path,
    add_blank=True,
    datasets=[dataset_config],
    #    characters=characters,
    enable_eos_bos_chars=True,
    mixed_precision=False,
    save_step=10000,
    save_n_checkpoints=2,
    save_best_after=5000,
    text_cleaner="no_cleaners",
    audio=audio_config,
    test_sentences=[],
    use_phonemes=True,
    phoneme_language="be",
)

if __name__ == "__main__":
    # INITIALIZE THE AUDIO PROCESSOR
    # Audio processor is used for feature extraction and audio I/O.
    # It mainly serves to the dataloader and the training loggers.
    ap = AudioProcessor.init_from_config(config)

    # INITIALIZE THE TOKENIZER
    # Tokenizer is used to convert text to sequences of token IDs.
    # If characters are not defined in the config, default characters are passed to the config
    tokenizer, config = TTSTokenizer.init_from_config(config)

    # LOAD DATA SAMPLES
    # Each sample is a list of ```[text, audio_file_path, speaker_name]```
    # You can define your custom sample loader returning the list of samples.
    # Or define your custom formatter and pass it to the `load_tts_samples`.
    # Check `TTS.tts.datasets.load_tts_samples` for more details.
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    # INITIALIZE THE MODEL
    # Models take a config object and a speaker manager as input
    # Config defines the details of the model like the number of layers, the size of the embedding, etc.
    # Speaker manager is used by multi-speaker models.
    model = GlowTTS(config, ap, tokenizer, speaker_manager=None)

    # INITIALIZE THE TRAINER
    # Trainer provides a generic API to train all the 🐸TTS models with all its perks like mixed-precision training,
    # distributed training, etc.
    trainer = Trainer(
        TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
    )

    # AND... 3,2,1... 🚀
    trainer.fit()
