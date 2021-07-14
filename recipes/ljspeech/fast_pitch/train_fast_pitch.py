import os

from TTS.config import BaseAudioConfig, BaseDatasetConfig
from TTS.trainer import Trainer, TrainingArgs, init_training
from TTS.tts.configs import FastPitchConfig

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(name="ljspeech", meta_file_train="metadata.csv",  meta_file_attn_mask=os.path.join(output_path, "../LJSpeech-1.1/metadata_attn_mask.txt"), path=os.path.join(output_path, "../LJSpeech-1.1/"))
audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=False,
    trim_db=0.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)
config = FastPitchConfig(
    run_name="fast_pitch_ljspeech",
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=4,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config]
)
args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
trainer = Trainer(args, config, output_path, c_logger, tb_logger)
trainer.fit()
