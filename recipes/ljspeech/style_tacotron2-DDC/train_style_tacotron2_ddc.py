import os

# Vanilla Imports
from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.tts.models.styletacotron2 import StyleTacotron2
from TTS.tts.configs.styletacotron2_config import StyleTacotron2Config

# Style Encoder Imports (Config and Layer)
from TTS.style_encoder.configs.style_encoder_config import StyleEncoderConfig

output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "../LJSpeech-1.1/")
)

audio_config = BaseAudioConfig(
    sample_rate=22050,
    do_trim_silence=True,
    trim_db=60.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

# Define Encoder Type and Parameters
style_config = StyleEncoderConfig()

# Pass the Encoder Config to the Model Config
config = StyleTacotronConfig( 
    style_encoder_config=style_config,
    num_eval_loader_workers=0,
    use_noise_augment=False,
    audio=audio_config,
    batch_size=32,
    batch_group_size=4,
    min_seq_len=2,
    max_seq_len=400,
    eval_batch_size=1,
    num_loader_workers=4,
    run_eval=True,
    test_delay_epochs=10,
    r=6,
    gradual_training=[[0, 6, 64], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=False,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    scheduler_after_epoch=False,
    model_param_stats=False,
    save_step=5000,
    checkpoint=True,
    keep_all_best=False,
    keep_after=10000,
    plot_step=100,

)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init model
model = StyleTacotron2(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()