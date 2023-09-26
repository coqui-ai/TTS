from trainer import Trainer, TrainerArgs

from TTS.tts.configs.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs.hifigan_config import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = "/storage/output-hifigan/"

audio_config = BaseAudioConfig(
    mel_fmin=50,
    mel_fmax=8000,
    hop_length=256,
    stats_path="/storage/TTS/scale_stats.npy",
)

config = HifiganConfig(
    batch_size=74,
    eval_batch_size=16,
    num_loader_workers=8,
    num_eval_loader_workers=8,
    lr_disc=0.0002,
    lr_gen=0.0002,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    use_noise_augment=True,
    seq_len=8192,
    pad_short=2000,
    save_step=5000,
    print_step=50,
    print_eval=True,
    mixed_precision=False,
    eval_split_size=30,
    save_n_checkpoints=2,
    save_best_after=5000,
    data_path="/storage/filtered_dataset",
    output_path=output_path,
    audio=audio_config,
)

# init audio processor
ap = AudioProcessor.init_from_config(config)

# load training samples
print("config.eval_split_size = ", config.eval_split_size)
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
