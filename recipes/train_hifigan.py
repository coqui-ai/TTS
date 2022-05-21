import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.enhancer.config.hifigan_config import HifiganConfig
from TTS.enhancer.models.hifigan import HifiGAN, HifiGANArgs
from TTS.enhancer.datasets.dataset import load_wav_data
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname("/home/julian/workspace/train")

audio_config = BaseAudioConfig(
    sample_rate=16000,
    num_mels=80,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=True,
    trim_db=15,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
    resample=False,
)

bweArgs = HifiGANArgs()

config = HifiganConfig(
    model_args=bweArgs,
    audio=audio_config,
    run_name="enhancer_hifigan_vctk",
    batch_size=6,
    eval_batch_size=6,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    target_sr=16000,
    input_sr=16000,
    gt_augment=True,
    segment_train=True,
    segment_len=2,
    cudnn_benchmark=False,
    epochs=25,
    print_step=25,
    save_step=10000,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=["/media/julian/Datasets/VCTK-VAD-16kHz/"],
    audio_augmentation={
        "p": 1,
        "additive": {
            "sounds_path": "/media/julian/Workdisk/datasets/DNS-Challenge/",
            "noise": {"min_snr_in_db": 15, "max_snr_in_db": 35, "min_num_noises": 1, "max_num_noises": 1},
        },
        "rir": { 
            "rir_path": "/media/julian/Datasets/RIRS_NOISES",
            "conv_mode": "full"
        },
        "EQ": {
            "min_snr_in_db": -10,
            "max_snr_in_db": 10,
            "p": 0.8
        },
        "gaussian": {
            "min_snr_in_db": 25,
            "max_snr_in_db": 45,
            "p": 0.3
        }
    },
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

#
train_samples, eval_samples = load_wav_data(config.datasets, eval_split_size=config.eval_split_size)

# init model
model = HifiGAN(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
    train_samples=train_samples,
    eval_samples=eval_samples,
)

trainer.fit()
