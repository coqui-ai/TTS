import glob
import os
import shutil

from tests import get_device_id, get_tests_output_path, run_cli
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig

config_path = os.path.join(get_tests_output_path(), "test_model_config.json")
output_path = os.path.join(get_tests_output_path(), "train_outputs")


dataset_config_en = BaseDatasetConfig(
    name="ljspeech",
    meta_file_train="metadata.csv",
    meta_file_val="metadata.csv",
    path="tests/data/ljspeech",
    language="en",
)

dataset_config_pt = BaseDatasetConfig(
    name="ljspeech",
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
    text_cleaner="english_cleaners",
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
    datasets=[dataset_config_en, dataset_config_pt],
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
config.model_args.d_vector_file = "tests/data/ljspeech/speakers.json"
config.d_vector_file = "tests/data/ljspeech/speakers.json"
config.model_args.d_vector_dim = 256
config.d_vector_dim = 256

# duration predictor
config.model_args.use_sdp = True
config.use_sdp = True

# deactivate language sampler
config.use_language_weighted_sampler = False

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

# restore the model and continue training for one more epoch
command_train = f"CUDA_VISIBLE_DEVICES='{get_device_id()}' python TTS/bin/train_tts.py --continue_path {continue_path} "
run_cli(command_train)
shutil.rmtree(continue_path)
