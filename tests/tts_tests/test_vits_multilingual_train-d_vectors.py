import glob
import json
import os
import shutil

from trainer import get_last_checkpoint

from tests import get_device_id, get_tests_output_path, run_cli
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")


dataset_config_en = BaseDatasetConfig(
    formatter="ljspeech_test",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="en",
)

dataset_config_pt = BaseDatasetConfig(
    formatter="ljspeech_test",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="pt-br",
)

config = VitsConfig(
    batch_size=2,
    eval_batch_size=2,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    text_cleaner="multilingual_cleaners",
    use_phonemes=False,
    phoneme_cache_path="tests/data/ljspeech/phoneme_cache/",
    run_eval=True,
    test_delay_epochs=-1,
    epochs=1,
    print_step=1,
    print_eval=True,
    test_sentences=[
        ["Be a voice, not an echo.", "ljspeech-0", None, "en"],
        ["Be a voice, not an echo.", "ljspeech-1", None, "pt-br"],
    ],
    datasets=[dataset_config_en, dataset_config_en, dataset_config_en, dataset_config_pt],
)
# set audio config
config.audio.do_trim_silence = True
config.audio.trim_db = 60

# active multilingual mode
config.model_args.use_language_embedding = True
config.use_language_embedding = True

# deactivate multispeaker mode
config.model_args.use_speaker_embedding = False
config.use_speaker_embedding = False

# active multispeaker d-vec mode
config.model_args.use_d_vector_file = True
config.use_d_vector_file = True
config.model_args.d_vector_file = ["tests/data/ljspeech/speakers.json"]
config.d_vector_file = ["tests/data/ljspeech/speakers.json"]
config.model_args.d_vector_dim = 256
config.d_vector_dim = 256

# duration predictor
config.model_args.use_sdp = True
config.use_sdp = True

# activate language and speaker samplers
config.use_language_weighted_sampler = True
config.language_weighted_sampler_alpha = 10
config.use_speaker_weighted_sampler = True
config.speaker_weighted_sampler_alpha = 5

config.save_json(config_path)

# train the model for one epoch
command_train = (
    f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --config_path {config_path} "
    f"--coqpit.output_path {output_path} "
    "--coqpit.test_delay_epochs 0"
)
run_cli(command_train)

# Find latest folder
continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)

# Inference using TTS API
continue_config_path = os.path.join(continue_path, "config.json")
continue_restore_path, _ = get_last_checkpoint(continue_path)
out_wav_path = os.path.join(get_tests_output_path(), "output.wav")
speaker_id = "ljspeech-1"
languae_id = "en"
continue_speakers_path = config.d_vector_file
continue_languages_path = os.path.join(continue_path, "language_ids.json")

# Check integrity of the config
with open(continue_config_path, "r", encoding="utf-8") as f:
    config_loaded = json.load(f)
assert config_loaded["characters"] is not None
assert config_loaded["output_path"] in continue_path
assert config_loaded["test_delay_epochs"] == 0

# Load the model and run inference
inference_command = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' tts --text 'This is an example.' --speaker_idx {speaker_id} --speakers_file_path {continue_speakers_path} --language_ids_file_path {continue_languages_path} --language_idx {languae_id} --config_path {continue_config_path} --model_path {continue_restore_path} --out_path {out_wav_path}"
run_cli(inference_command)

# restore the model and continue training for one more epoch
command_train = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)
