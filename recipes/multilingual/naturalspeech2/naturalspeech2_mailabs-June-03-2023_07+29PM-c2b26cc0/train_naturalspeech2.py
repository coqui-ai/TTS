import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.naturalspeech2_config import Naturalspeech2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.naturalspeech2 import Naturalspeech2, Naturalspeech2Args, Naturalspeech2AudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

# dataset_paths = glob(mailabs_path)
# init configs
dataset_config = BaseDatasetConfig(
    formatter="ljspeech",
    meta_file_train="metadata.csv",
    # meta_file_attn_mask=os.path.join(output_path, "../LJSpeech-1.1/metadata_attn_mask.txt"),
    path=os.path.join(output_path, "../../ljspeech/LJSpeech-1.1/"),
)

audio_config = Naturalspeech2AudioConfig(
    sample_rate=16000,
    win_length=1024,
    hop_length=256,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

naturalspeech2Args = Naturalspeech2Args(segment_size=32)

config = Naturalspeech2Config(
    model_args=naturalspeech2Args,
    audio=audio_config,
    run_name="naturalspeech2_mailabs",
    batch_size=2,
    eval_batch_size=2,
    batch_group_size=0,
    num_loader_workers=12,
    num_eval_loader_workers=12,
    precompute_num_workers=12,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en",
    phonemizer="espeak",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_language_weighted_sampler=True,
    print_eval=False,
    mixed_precision=False,
    min_audio_len=audio_config.sample_rate,
    max_audio_len=audio_config.sample_rate * 10,
    output_path=output_path,
    datasets=dataset_config,
)
# dict_ = config.to_dict()
# print(dict_)
# force the convertion of the custom characters to a config attribute
# config.from_dict(config.to_dict())

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# language_manager = LanguageManager(config=config)
# config.model_args.num_languages = language_manager.num_languages

# INITIALIZE THE TOKENIZER
# Tokenizer is used to convert text to sequences of token IDs.
# config is updated with the default characters if not defined in the config.
tokenizer, config = TTSTokenizer.init_from_config(config)

# init model
model = Naturalspeech2(config, ap, tokenizer)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
