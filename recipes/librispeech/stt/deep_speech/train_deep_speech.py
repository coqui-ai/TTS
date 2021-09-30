import os

from TTS.config.shared_configs import BaseAudioConfig, BaseDatasetConfig
from TTS.stt.configs import DeepSpeechConfig
from TTS.stt.datasets import load_stt_samples
from TTS.stt.datasets.downloaders import download_librispeech
from TTS.stt.datasets.tokenizer import Tokenizer
from TTS.stt.models.deep_speech import DeepSpeech
from TTS.trainer import Trainer, TrainingArgs
from TTS.utils.audio import AudioProcessor

output_path = os.path.dirname(os.path.abspath(__file__))

if not os.path.exists("/home/ubuntu/librispeech/LibriSpeech/train-clean-100"):
    download_librispeech("/home/ubuntu/librispeech/", "train-clean-100")
if not os.path.exists("/home/ubuntu/librispeech/LibriSpeech/dev-clean"):
    download_librispeech("/home/ubuntu/librispeech/", "dev-clean")

# train_dataset_config = BaseDatasetConfig(
# name="librispeech", meta_file_train=None, path="/home/ubuntu/librispeech/LibriSpeech/train-clean-100"
# )

# eval_dataset_config = BaseDatasetConfig(
# name="librispeech", meta_file_train=None, path="/home/ubuntu/librispeech/LibriSpeech/dev-clean"
# )

train_dataset_config = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.csv",
    path="/home/ubuntu/ljspeech/LJSpeech-1.1/",
)


audio_config = BaseAudioConfig(
    sample_rate=16000,
    resample=False,
    frame_shift_ms=20,
    frame_length_ms=40,
    num_mels=26,
    num_mfcc=26,
    do_trim_silence=False,
    mel_fmin=0,
    mel_fmax=8000,
)

config = DeepSpeechConfig(
    audio=audio_config,
    run_name="deepspeech_librispeech",
    batch_size=128,
    eval_batch_size=16,
    batch_group_size=5,
    num_loader_workers=4,
    num_eval_loader_workers=4,
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    print_step=25,
    print_eval=False,
    mixed_precision=True,
    max_seq_len=500000,
    output_path=output_path,
    train_datasets=[train_dataset_config],
    # eval_datasets=[eval_dataset_config],
)

# init audio processor
ap = AudioProcessor(**config.audio.to_dict())

# load training samples
train_samples, eval_samples = load_stt_samples(train_dataset_config, eval_split=True)
# eval_samples, _ = load_stt_samples(eval_dataset_config, eval_split=False)
transcripts = [s["text"] for s in train_samples]

# init tokenizer
tokenizer = Tokenizer(transcripts=transcripts)
n_tokens = tokenizer.vocab_size
config.model_args.n_tokens = n_tokens
config.vocabulary = tokenizer.vocab_dict

# init model
model = DeepSpeech(config)

# init training and kick it 🚀
# args, config, output_path, _, c_logger, tb_logger = init_training(TrainingArgs(), config)
trainer = Trainer(
    TrainingArgs(),
    config,
    output_path,
    model=model,
    tokenizer=tokenizer,
    train_samples=train_samples,
    eval_samples=eval_samples,
    cudnn_benchmark=False,
    training_assets={"tokenizer": tokenizer, "audio_processor": ap},
)
trainer.fit()
