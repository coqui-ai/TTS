import glob
import json
import os
import shutil

from trainer import get_last_checkpoint

from tests import get_device_id, get_tests_output_path, run_cli
from TTS.tts.configs.delightful_tts_config import DelightfulTtsAudioConfig, DelightfulTTSConfig
from TTS.tts.models.delightful_tts import DelightfulTtsArgs, VocoderConfig

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")


audio_config = DelightfulTtsAudioConfig()
model_args = DelightfulTtsArgs(
    use_speaker_embedding=False, d_vector_dim=256, use_d_vector_file=True, speaker_embedding_channels=256
)

vocoder_config = VocoderConfig()

config = DelightfulTTSConfig(
    model_args=model_args,
    audio=audio_config,
    vocoder=vocoder_config,
    batch_size=2,
    eval_batch_size=8,
    compute_f0=True,
    run_eval=True,
    test_delay_epochs=-1,
    text_cleaner="english_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path="tests/data/ljspeech/phoneme_cache/",
    f0_cache_path="tests/data/ljspeech/f0_cache_delightful/",  ## delightful f0 cache is incompatible with other models
    epochs=1,
    print_step=1,
    print_eval=True,
    binary_align_loss_alpha=0.0,
    use_attn_priors=False,
    test_sentences=[
        ["Be a voice, not an echo.", "ljspeech-0"],
    ],
    output_path=output_path,
    use_speaker_embedding=False,
    use_d_vector_file=True,
    d_vector_file="tests/data/ljspeech/speakers.json",
    d_vector_dim=256,
    speaker_embedding_channels=256,
)

# active multispeaker d-vec mode
config.model_args.use_speaker_embedding = False
config.model_args.use_d_vector_file = True
config.model_args.d_vector_file = "tests/data/ljspeech/speakers.json"
config.model_args.d_vector_dim = 256


config.save_json(config_path)

command_train = (
    f"CUDA_VISIBLE_DEVICES='{get_device_id()}'  python TTS/bin/train_tts.py --config_path {config_path}  "
    f"--coqpit.output_path {output_path} "
    "--coqpit.datasets.0.formatter ljspeech "
    "--coqpit.datasets.0.meta_file_train metadata.csv "
    "--coqpit.datasets.0.meta_file_val metadata.csv "
    "--coqpit.datasets.0.path tests/data/ljspeech "
    "--coqpit.datasets.0.meta_file_attn_mask tests/data/ljspeech/metadata_attn_mask.txt "
    "--coqpit.test_delay_epochs 0"
)

run_cli(command_train)

# Find latest folder
continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)

# Inference using TTS API
continue_config_path = os.path.join(continue_path, "config.json")
continue_restore_path, _ = get_last_checkpoint(continue_path)
speaker_id = "ljspeech-1"
continue_speakers_path = config.d_vector_file

out_wav_path = os.path.join(get_tests_output_path(), "output.wav")
# Check integrity of the config
with open(continue_config_path, "r", encoding="utf-8") as f:
    config_loaded = json.load(f)
assert config_loaded["characters"] is not None
assert config_loaded["output_path"] in continue_path
assert config_loaded["test_delay_epochs"] == 0

# Load the model and run inference
inference_command = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' tts --text 'This is an example.' --speaker_idx {speaker_id} --config_path {continue_config_path} --speakers_file_path {continue_speakers_path} --model_path {continue_restore_path} --out_path {out_wav_path}"
run_cli(inference_command)

# restore the model and continue training for one more epoch
command_train = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)
shutil.rmtree("tests/data/ljspeech/f0_cache_delightful/")
