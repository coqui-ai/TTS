import os

from TTS.config.shared_configs import BaseAudioConfig
from TTS.trainer import Trainer, TrainingArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron_config import TacotronConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron import Tacotron
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))
dataset_config = BaseDatasetConfig(name="vctk", meta_file_train="", path=os.path.join(output_path, "../VCTK/"))

audio_config = BaseAudioConfig(
    sample_rate=22050,
    resample=True,  # Resample to 22050 Hz. It slows down training. Use `TTS/bin/resample.py` to pre-resample and set this False for faster training.
    do_trim_silence=True,
    trim_db=23.0,
    signal_norm=False,
    mel_fmin=0.0,
    mel_fmax=8000,
    spec_gain=1.0,
    log_func="np.log",
    ref_level_db=20,
    preemphasis=0.0,
)

config = TacotronConfig(  # This is the config that is saved for the future use
    audio=audio_config,
    batch_size=48,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    r=6,
    gradual_training=[[0, 6, 48], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=True,
    epochs=1000,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    sort_by_audio_len=True,
    min_seq_len=0,
    max_seq_len=44000 * 10,  # 44k is the original sampling rate before resampling, corresponds to 10 seconds of audio
    output_path=output_path,
    datasets=[dataset_config],
    use_speaker_embedding=True,  # set this to enable multi-sepeaker training
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init speaker manager for multi-speaker training
# it mainly handles speaker-id to speaker-name for the model and the data-loader
speaker_manager = SpeakerManager()
speaker_manager.set_speaker_ids_from_data(train_samples + eval_samples)

# init model
model = Tacotron(config, speaker_manager)

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
