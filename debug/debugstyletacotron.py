import os
import time
from TTS.config.shared_configs import BaseAudioConfig
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.tacotron2_config import Tacotron2Config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.tacotron2 import Tacotron2
from TTS.utils.audio import AudioProcessor
from TTS.trainer import Trainer, TrainingArgs

# from TTS.tts.datasets.tokenizer import Tokenizer

output_path = os.path.dirname(os.path.abspath(__file__))

# init configs
dataset_config = BaseDatasetConfig(
    name="ljspeech", meta_file_train="metadata.csv", path=os.path.join(output_path, "C:/Users/leona/Documents/Git Repositories/AI-Unicamp/TTS/recipes/ljspeech/LJSpeech-1.1")
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

config = Tacotron2Config(  # This is the config that is saved for the future use
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

# initialize the data loader
trainer.train_loader = trainer.get_train_dataloader(trainer.training_assets,trainer.train_samples, verbose=True)
# set model to training mode
if trainer.num_gpus > 1:
    trainer.model.module.train()
else:
    trainer.model.train()
epoch_start_time = time.time()
if trainer.use_cuda:
    batch_num_steps = int(len(trainer.train_loader.dataset) / (trainer.config.batch_size * trainer.num_gpus))
else:
    batch_num_steps = int(len(trainer.train_loader.dataset) / trainer.config.batch_size)
trainer.c_logger.print_train_start()
loader_start_time = time.time()
# iterate over the training samples
batch = next(x for x in trainer.train_loader)
print(batch)
"""
from TTS.tts.models.styletacotron2 import StyleTacotron2
from TTS.tts.configs.style_tacotron2_config import StyleTacotronConfig
from TTS.style_encoder.configs.style_encoder_config import StyleEncoderConfig

config = StyleTacotronConfig(
    style_encoder_config= StyleEncoderConfig()
)

a = StyleTacotron2(config)

B = 32
T_in = 95
T_out = 5*T_in
C = 80

text = torch.randint(1, 50, (B, T_in))
text_lengths = torch.randint(1, T_in, (B,))
mel_specs = torch.rand(B, T_out, C)
mel_lengths = torch.randint(1, T_out, (B,))

print(text_lengths.size())
#a(text, text_lengths, mel_specs, mel_lengths)
"""