import glob
import os
import shutil

from tests import get_device_id, get_tests_output_path, run_cli
from TTS.config.shared_configs import BaseAudioConfig
from TTS.speaker_encoder.speaker_encoder_config import SpeakerEncoderConfig


def run_test_train():
    command = (
        f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_encoder.py --config_path {config_path} "
        f"--coqpit.output_path {output_path} "
        "--coqpit.datasets.0.name ljspeech "
        "--coqpit.datasets.0.meta_file_train metadata.csv "
        "--coqpit.datasets.0.meta_file_val metadata.csv "
        "--coqpit.datasets.0.path tests/data/ljspeech "
    )
    run_cli(command)


config_path = os.path.join(get_tests_output_path(), "test_speaker_encoder_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")

config = SpeakerEncoderConfig(
    batch_size=4,
    num_speakers_in_batch=1,
    num_utters_per_speaker=10,
    num_loader_workers=0,
    max_train_step=2,
    print_step=1,
    save_step=1,
    print_eval=True,
    audio=BaseAudioConfig(num_mels=80),
)
config.audio.do_trim_silence = True
config.audio.trim_db = 60
config.save_json(config_path)

print(config)
# train the model for one epoch
run_test_train()

# Find latest folder
continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)

# restore the model and continue training for one more epoch
command_train = (
    f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_encoder.py --continue_path {continue_path} "
)
run_cli(command_train)
shutil.rmtree(continue_path)

# test resnet speaker encoder
config.model_params["model_name"] = "resnet"
config.save_json(config_path)

# train the model for one epoch
run_test_train()

# Find latest folder
continue_path = max(glob.glob(os.path.join(output_path, "*/")), key=os.path.getmtime)

# restore the model and continue training for one more epoch
command_train = (
    f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_encoder.py --continue_path {continue_path} "
)
run_cli(command_train)
shutil.rmtree(continue_path)

# test model with ge2e loss function
config.loss = "ge2e"
config.save_json(config_path)
run_test_train()

# test model with angleproto loss function
config.loss = "angleproto"
config.save_json(config_path)
run_test_train()

# test model with softmaxproto loss function
config.loss = "softmaxproto"
config.save_json(config_path)
run_test_train()
