import os

from trainer import Trainer, TrainerArgs

from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import WavernnConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.wavernn import Wavernn

output_path = os.path.dirname(os.path.abspath(__file__))
config = WavernnConfig(
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=10000,
    seq_len=1280,
    pad_short=2000,
    use_noise_augment=False,
    eval_split_size=10,
    print_step=25,
    print_eval=True,
    mixed_precision=False,
    lr=1e-4,
    grad_clip=4,
    data_path=os.path.join(output_path, "../LJSpeech-1.1/wavs/"),
    output_path=output_path,
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size)

# init model
model = Wavernn(config)

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
