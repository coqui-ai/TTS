#!/usr/bin/env python3`
import glob
import os
import shutil

from tests import get_tests_data_path, get_tests_output_path, run_cli
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

MODELS_WITH_SEP_TESTS = ["bark", "xtts"]


def run_models(offset=0, step=1):
    """Check if all the models are downloadable and tts models run correctly."""
    print(" > Run synthesizer with all the models.")
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    manager = ModelManager(output_prefix=get_tests_output_path(), progress_bar=False)
    model_names = [name for name in manager.list_models() if name in MODELS_WITH_SEP_TESTS]
    for model_name in model_names[offset::step]:
        print(f"\n > Run - {model_name}")
        model_path, _, _ = manager.download_model(model_name)
        if "tts_models" in model_name:
            local_download_dir = os.path.dirname(model_path)
            # download and run the model
            speaker_files = glob.glob(local_download_dir + "/speaker*")
            language_files = glob.glob(local_download_dir + "/language*")
            language_id = ""
            if len(speaker_files) > 0:
                # multi-speaker model
                if "speaker_ids" in speaker_files[0]:
                    speaker_manager = SpeakerManager(speaker_id_file_path=speaker_files[0])
                elif "speakers" in speaker_files[0]:
                    speaker_manager = SpeakerManager(d_vectors_file_path=speaker_files[0])

                # multi-lingual model - Assuming multi-lingual models are also multi-speaker
                if len(language_files) > 0 and "language_ids" in language_files[0]:
                    language_manager = LanguageManager(language_ids_file_path=language_files[0])
                    language_id = language_manager.language_names[0]

                speaker_id = list(speaker_manager.name_to_id.keys())[0]
                run_cli(
                    f"tts --model_name  {model_name} "
                    f'--text "This is an example." --out_path "{output_path}" --speaker_idx "{speaker_id}" --language_idx "{language_id}" --progress_bar False'
                )
            else:
                # single-speaker model
                run_cli(
                    f"tts --model_name  {model_name} "
                    f'--text "This is an example." --out_path "{output_path}" --progress_bar False'
                )
            # remove downloaded models
            shutil.rmtree(local_download_dir)
            shutil.rmtree(get_user_data_dir("tts"))
        elif "voice_conversion_models" in model_name:
            speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
            reference_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav")
            run_cli(
                f"tts --model_name  {model_name} "
                f'--out_path "{output_path}" --source_wav "{speaker_wav}" --target_wav "{reference_wav}" --progress_bar False'
            )
        else:
            # only download the model
            manager.download_model(model_name)
        print(f" | > OK: {model_name}")


def test_xtts():
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    run_cli(
        "yes | "
        f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1 "
        f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True '
        f'--speaker_wav "{speaker_wav}" --language_idx "en"'
    )


def test_bark():
    """Bark is too big to run on github actions. We need to test it locally"""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    run_cli(
        f" tts --model_name  tts_models/multilingual/multi-dataset/bark "
        f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True'
    )


def test_voice_conversion():
    print(" > Run voice conversion inference using YourTTS model.")
    model_name = "tts_models/multilingual/multi-dataset/your_tts"
    language_id = "en"
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    reference_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav")
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    run_cli(
        f"tts --model_name  {model_name}"
        f" --out_path {output_path} --speaker_wav {speaker_wav} --reference_wav {reference_wav} --language_idx {language_id} --progress_bar False"
    )


"""
These are used to split tests into different actions on Github.
"""


def test_models_offset_0_step_3():
    run_models(offset=0, step=3)


def test_models_offset_1_step_3():
    run_models(offset=1, step=3)


def test_models_offset_2_step_3():
    run_models(offset=2, step=3)
