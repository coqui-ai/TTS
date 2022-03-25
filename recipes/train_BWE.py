import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.enhancer.config.base_enhancer_config import BaseEnhancerConfig
from TTS.enhancer.models.bwe import BWE, BWEArgs
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname("/home/julian/workspace/train")

dataset_config = BaseDatasetConfig(
    name="vctk", meta_file_train="", language="en-us", path="/media/julian/Workdisk/datasets/VCTK-Corpus-48"
)

audio_config = BaseAudioConfig(
    sample_rate=48000,
    num_mels=128,
    preemphasis=0.0,
    ref_level_db=20,
    log_func="np.log",
    do_trim_silence=False,
    mel_fmin=0,
    mel_fmax=None,
    spec_gain=1.0,
    signal_norm=False,
    do_amp_to_db_linear=False,
    resample=False,
)

bweArgs = BWEArgs(
    num_channel_wn=128,
    dilation_rate_wn=3,
    num_blocks_wn=2,
    num_layers_wn=7,
)

config = BaseEnhancerConfig(
    model_args=bweArgs,
    audio=audio_config,
    run_name="enhancer_bwe_vctk",
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    print_step=25,
    print_eval=False,
    mixed_precision=False,
    output_path=output_path,
    datasets=[dataset_config],
    audio_augmentation={
        "data_augmentation_p":0.5,
        "additive":{
            "sounds_path": "/media/julian/Workdisk/datasets/musan/noise"
        }
    }
)

# INITIALIZE THE AUDIO PROCESSOR
# Audio processor is used for feature extraction and audio I/O.
# It mainly serves to the dataloader and the training loggers.
ap = AudioProcessor.init_from_config(config)

# init model
model = BWE(config, ap)

# init the trainer and 🚀
trainer = Trainer(
    TrainerArgs(),
    config,
    output_path,
    model=model,
)
trainer.fit()
