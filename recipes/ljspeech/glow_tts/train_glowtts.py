import os

from TTS.trainer import Trainer, TrainingArgs, init_training
from TTS.tts.configs import BaseDatasetConfig, GlowTTSConfig

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(
    name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "../LJSpeech-1.1/")
)
config = GlowTTSConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=False,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
)
args, config, output_path, _, c_logger, dashboard_logger = init_training(TrainingArgs(), config)
trainer = Trainer(args, config, output_path, c_logger, dashboard_logger)
trainer.fit()
