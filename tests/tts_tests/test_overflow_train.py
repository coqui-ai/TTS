import glob
import json
import os
import shutil

import torch
from trainer import get_last_checkpoint

from tests import get_device_id, get_tests_output_path, run_cli
from TTS.tts.configs.overflow_config import OverflowConfig

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")
parameter_path = os.path.join(get_tests_output_path(), "lj_parameters.pt")

torch.save({"mean": -5.5138, "std": 2.0636, "init_transition_prob": 0.3212}, parameter_path)

config = OverflowConfig(
    batch_size=3,
    eval_batch_size=3,
    num_loader_workers=0,
    num_eval_loader_workers=0,
    text_cleaner="phoneme_cleaners",
    use_phonemes=True,
    phoneme_language="en-us",
    phoneme_cache_path=os.path.join(get_tests_output_path(), "train_outputs/phoneme_cache/"),
    run_eval=True,
    test_delay_epochs=-1,
    mel_statistics_parameter_path=parameter_path,
    epochs=1,
    print_step=1,
    test_sentences=[
        "Be a voice, not an echo.",
    ],
    print_eval=True,
    max_sampling_time=50,
)
config.audio.do_trim_silence = True
config.audio.trim_db = 60
config.save_json(config_path)


# train the model for one epoch when mel parameters exists
command_train = (
    f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --config_path {config_path} "
    f"--coqpit.output_path {output_path} "
    "--coqpit.datasets.0.formatter ljspeech "
    "--coqpit.datasets.0.meta_file_train metadata.csv "
    "--coqpit.datasets.0.meta_file_val metadata.csv "
    "--coqpit.datasets.0.path tests/data/ljspeech "
    "--coqpit.test_delay_epochs 0 "
)
run_cli(command_train)


# train the model for one epoch when mel parameters have to be computed from the dataset
if os.path.exists(parameter_path):
    os.remove(parameter_path)
command_train = (
    f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --config_path {config_path} "
    f"--coqpit.output_path {output_path} "
    "--coqpit.datasets.0.formatter ljspeech "
    "--coqpit.datasets.0.meta_file_train metadata.csv "
    "--coqpit.datasets.0.meta_file_val metadata.csv "
    "--coqpit.datasets.0.path tests/data/ljspeech "
    "--coqpit.test_delay_epochs 0 "
)
run_cli(command_train)

# Find latest folder
continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)

# Inference using TTS API
continue_config_path = os.path.join(continue_path, "config.json")
continue_restore_path, _ = get_last_checkpoint(continue_path)
out_wav_path = os.path.join(get_tests_output_path(), "output.wav")

# Check integrity of the config
with open(continue_config_path, "r", encoding="utf-8") as f:
    config_loaded = json.load(f)
assert config_loaded["characters"] is not None
assert config_loaded["output_path"] in continue_path
assert config_loaded["test_delay_epochs"] == 0

# Load the model and run inference
inference_command = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' tts --text 'This is an example.' --config_path {continue_config_path} --model_path {continue_restore_path} --out_path {out_wav_path}"
run_cli(inference_command)

# restore the model and continue training for one more epoch
command_train = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)
