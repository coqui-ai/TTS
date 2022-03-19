import os
import time
import torch
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig, GSTConfig
from TTS.tts.datasets import load_tts_samples
from TTS.utils.audio import AudioProcessor
from TTS.trainer import Trainer, TrainingArgs

# Old Tacotron Imports
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.configs.shared_configs import GSTConfig

# Style Tacotron Imports
from TTS.tts.models.styletacotron2 import StyleTacotron2
from TTS.tts.configs.style_tacotron2_config import StyleTacotronConfig
from TTS.style_encoder.configs.style_encoder_config import StyleEncoderConfig

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(42)
output_path = os.path.dirname(os.path.abspath(os.path.abspath('')))

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "/home/leonardo/Documentos/Git Repositories/AI-Unicamp/TTS/recipes/ljspeech/LJSpeech-1.1")
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

gst_config = GSTConfig()

config = Tacotron2Config(  # This is the config that is saved for the future use
    use_gst = True,
    gst = gst_config,
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    r=6,
    gradual_training=[[0, 6, 64], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=True,
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
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_tts_samples(dataset_config, eval_split=True)

# init model
model = Tacotron2(config)

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

# Data loader
trainer.train_loader = trainer.get_train_dataloader(trainer.training_assets,trainer.train_samples, verbose=True)
# iterate over the training samples
batch = trainer.format_batch(next(x for x in trainer.train_loader))
batch.keys()
input_args = [batch, trainer.criterion]
old_outs, old_loss = trainer.model.train_step(*input_args)

seed_everything(42)
output_path = os.path.dirname(os.path.abspath(os.path.abspath('')))

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "/home/leonardo/Documentos/Git Repositories/AI-Unicamp/TTS/recipes/ljspeech/LJSpeech-1.1")
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

style_config = StyleEncoderConfig(se_type="gst")

config = StyleTacotronConfig(  # This is the config that is saved for the future use
    style_encoder_config = style_config,
    audio=audio_config,
    batch_size=64,
    eval_batch_size=16,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    r=6,
    gradual_training=[[0, 6, 64], [10000, 4, 32], [50000, 3, 32], [100000, 2, 32]],
    double_decoder_consistency=True,
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
# Data loader
trainer.train_loader = trainer.get_train_dataloader(trainer.training_assets,trainer.train_samples, verbose=True)
# iterate over the training samples
batch = trainer.format_batch(next(x for x in trainer.train_loader))
batch.keys()
input_args = [batch, trainer.criterion]
new_outs , new_loss = trainer.model.train_step(*input_args)

#Finish
os.system('clear')
for key in new_outs.keys():
    old_val = old_outs[str(key)]
    new_val = new_outs[str(key)]
    print(str(key) + ' - ' + str(torch.equal(old_val, new_val)))