from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.utils.audio import AudioProcessor
from TTS.vocoder.configs import HifiganConfig
from TTS.vocoder.datasets.preprocess import load_wav_data
from TTS.vocoder.models.gan import GAN

output_path = "/home/julian/workspace/train"

audio_config = BaseAudioConfig(
    sample_rate=48000,
    num_mels=80,
    fft_size=2016,
    win_length=2016,
    hop_length=504,
    mel_fmax=8000,
    mel_fmin=0,
)

config = HifiganConfig(
    audio=audio_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=5,
    epochs=1000,
    seq_len=32 * audio_config.hop_length,
    generator_model_params={
        "upsample_factors": [8, 8, 4, 2],
        "upsample_kernel_sizes": [16, 16, 8, 4],
        "upsample_initial_channel": 512,
        "resblock_kernel_sizes": [3, 7, 11],
        "resblock_dilation_sizes": [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
        "resblock_type": "1",
    },
    l1_spec_loss_params={
        "use_mel": True,
        "sample_rate": 48000,
        "n_fft": 2048,
        "hop_length": 512,
        "win_length": 2048,
        "n_mels": 80,
        "mel_fmin": 0,
        "mel_fmax": 24000,
    },
    discriminator_model_params={"periods": [2, 3, 5, 7, 11, 17, 29]},
    pad_short=2000,
    use_noise_augment=True,
    eval_split_size=10,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    lr_gen=1e-4,
    lr_disc=1e-4,
    data_path="/media/julian/Workdisk/datasets/VCTK-Corpus-48/wav48_silence_trimmed/",
    output_path=output_path,
)

# init audio processor
ap = AudioProcessor.init_from_config(config)

# load training samples
eval_samples, train_samples = load_wav_data(config.data_path, config.eval_split_size, file_ext="flac")

# init model
model = GAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)
trainer.fit()
