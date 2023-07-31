import os

from trainer import Trainer, TrainerArgs

from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.delightful_tts_config import DelightfulTtsAudioConfig, DelightfulTTSConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.delightful_tts import DelightfulTTS, DelightfulTtsArgs, VocoderConfig
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio.processor import AudioProcessor

data_path = "/raid/datasets/vctk_v092_48khz_removed_silence_silero_vad"
output_path = os.path.dirname(os.path.abspath(__file__))


dataset_config = BaseDatasetConfig(
    dataset_name="vctk", formatter="vctk", meta_file_train="", path=data_path, language="en-us"
)

audio_config = DelightfulTtsAudioConfig()

model_args = DelightfulTtsArgs()

vocoder_config = VocoderConfig()

something_tts_config = DelightfulTTSConfig(
    run_name="delightful_tts_vctk",
    run_description="Train like in delightful tts paper.",
    model_args=model_args,
    audio=audio_config,
    vocoder=vocoder_config,
    batch_size=32,
    eval_batch_size=16,
    num_loader_workers=10,
    num_eval_loader_workers=10,
    precompute_num_workers=40,
    compute_input_seq_cache=True,
    compute_f0=True,
    f0_cache_path=os.path.join(output_path, "f0_cache"),
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1000,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
    print_step=50,
    print_eval=False,
    mixed_precision=True,
    output_path=output_path,
    datasets=[dataset_config],
    start_by_longest=True,
    binary_align_loss_alpha=0.0,
    use_attn_priors=False,
    max_text_len=60,
    steps_to_start_discriminator=10000,
)

tokenizer, config = TTSTokenizer.init_from_config(something_tts_config)

ap = AudioProcessor.init_from_config(config)


train_samples, eval_samples = load_tts_samples(
    dataset_config,
    eval_split=True,
    eval_split_max_size=config.eval_split_max_size,
    eval_split_size=config.eval_split_size,
)


speaker_manager = SpeakerManager()
speaker_manager.set_ids_from_data(train_samples + eval_samples, parse_key="speaker_name")
config.model_args.num_speakers = speaker_manager.num_speakers


model = DelightfulTTS(ap=ap, config=config, tokenizer=tokenizer, speaker_manager=speaker_manager, emotion_manager=None)

trainer = Trainer(
    TrainerArgs(), config, output_path, model=model, train_samples=train_samples, eval_samples=eval_samples
)

trainer.fit()
