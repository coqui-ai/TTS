import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.enhancer.configs.bwe_config import BweConfig
from TTS.enhancer.datasets.dataset import load_wav_data
from TTS.enhancer.models.bwe import Bwe, BweArgs
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname("/media/julian/Workdisk/train")

audio_config = BaseAudioConfig(
    sample_rate=48000,
    num_mels=128,
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

bweArgs = BweArgs(
    num_channel_wn=128,
    dilation_rate_wn=3,
    kernel_size_wn=3,
    num_blocks_wn=2,
    num_layers_wn=7,
)

config = BweConfig(
    model_args=bweArgs,
    audio=audio_config,
    run_name="enhancer_bwe_vctk",
    batch_size=4,
    eval_batch_size=4,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    print_step=25,
    save_step=10000,
    segment_train=True,
    segment_len=1,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=["/media/julian/Datasets/VCTK-VAD-16kHz/"],
    audio_augmentation={
        "p": 1,
        "additive": {
            "sounds_path": "/media/julian/Workdisk/datasets/DNS-Challenge/",
            "noise": {"min_snr_in_db": 15, "max_snr_in_db": 25, "min_num_noises": 1, "max_num_noises": 1},
        },
    },
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

#
train_samples, eval_samples = load_wav_data(config.datasets)

# init model
model = Bwe(config, ap)

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
