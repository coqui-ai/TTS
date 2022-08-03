import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.utils.downloaders import download_thorsten_de
from TTS.vocoder.configs import WavegradConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.wavegrad import Wavegrad

output_path = os.path.dirname(os.path.abspath(__file__))
config = WavegradConfig(
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    seq_len=6144,
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=50,
    print_step=50,
    print_eval=True,
    mixed_precision=False,
    data_path=os.path.join(output_path, "../thorsten-de/wavs/"),
    output_path=output_path,
)

# download dataset if not already present
if not os.path.exists(config.data_path):
    print("Downloading dataset")
    download_path = os.path.abspath(os.path.join(os.path.abspath(config.data_path), "../../"))
    download_thorsten_de(download_path)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = Wavegrad(config)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
    training_assets={"audio_processor": ap},
)
trainer.fit()
