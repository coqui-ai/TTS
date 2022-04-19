import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.fast_pitch_e2e_config import FastPitchE2eConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.forward_tts_e2e import ForwardTTSE2e, ForwardTTSE2eArgs, ForwardTTSE2eAudio
from TTS.tts.utils.text.tokenizer import TTSTokenizer

output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.csv",
    path=os.path.join(output_path, "../LJSpeech-1.1/"),
)

audio_config = ForwardTTSE2eAudio(
    sample_rate=22050,
    hop_length=256,
    win_length=1024,
    fft_size=1024,
    mel_fmin=0.0,
    mel_fmax=8000,
    pitch_fmax=640.0,
    num_mels=80,
)

# vocoder_config = HifiganConfig()
model_args = ForwardTTSE2eArgs()

config = FastPitchE2eConfig(
    run_name="fast_pitch_e2e_ljspeech",
    run_description="Train like in FS2 paper.",
    model_args=model_args,
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    precompute_num_workers=4,
    print_step=50,
    print_eval=False,
    mixed_precision=False,
    sort_by_audio_len=True,
    output_path=output_path,
    datasets=[dataset_config],
    start_by_longest=False,
    binary_align_loss_alpha=0.0,
)

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

# init the model
model = ForwardTTSE2e(config=config, tokenizer=tokenizer, speaker_manager=None)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
