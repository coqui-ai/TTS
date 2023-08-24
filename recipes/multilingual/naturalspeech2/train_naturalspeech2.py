import os
from glob import glob

from trainer import Trainer, TrainerArgs

from TTS.tts.configs.naturalspeech2_config import Naturalspeech2Config
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.naturalspeech2 import Naturalspeech2, Naturalspeech2Args, Naturalspeech2AudioConfig
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

# dataset_paths = glob(mailabs_path)
# init configs
dataset_config1 = BaseDatasetConfig(
    formatter="hifitts",
    meta_file_train=None,
    path="/root/Desktop/datasets/hifitts/hi_fi_tts_v0",
)

dataset_config2 = BaseDatasetConfig(
    formatter="vctk",
    meta_file_train=None,
    path="/root/Desktop/datasets/vctk",
)
libri_r_100 = BaseDatasetConfig(
    formatter="libri_r",
    meta_file_train=None,
    path="/root/Desktop/datasets/libritts_r/LibriTTS_R/train-clean-100",
)

libri_r_360 = BaseDatasetConfig(
    formatter="libri_r",
    meta_file_train=None,
    path="/root/Desktop/datasets/libritts_r/LibriTTS_R/train-clean-360",
)

libri_r_500 = BaseDatasetConfig(
    formatter="libri_r",
    meta_file_train=None,
    path="/root/Desktop/datasets/libritts_r/LibriTTS_R/train-other-500",
)
datasets = [libri_r_100, libri_r_360, libri_r_500, dataset_config1, dataset_config2]
# datasets = [libri_r_100]
audio_config = Naturalspeech2AudioConfig(
    sample_rate=24000,
    win_length=1024,
    hop_length=320,
    num_mels=80,
    mel_fmin=0,
    mel_fmax=None,
)

naturalspeech2Args = Naturalspeech2Args(diff_segment_size=300)

config = Naturalspeech2Config(
    model_args=naturalspeech2Args,
    audio=audio_config,
    run_name="naturalspeech2_mailabs",
    batch_size=24,
    eval_batch_size=8,
    batch_group_size=0,
    num_loader_workers=16,
    num_eval_loader_workers=16,
    precompute_num_workers=16,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en",
    phonemizer="espeak",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    compute_input_seq_cache=True,
    print_step=25,
    use_speaker_weighted_sampler=True,
    print_eval=False,
    mixed_precision=False,
    min_audio_len=audio_config.sample_rate,
    max_audio_len=audio_config.sample_rate * 10,
    output_path=output_path,
    datasets=datasets,
)


# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(
    datasets,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)

# speaker_manager = SpeakerManager()
# speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
# config.model_args.num_speakers = speaker_manager.num_speakers
# print("total number of speakers: ", speaker_manager.num_speakers)
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
