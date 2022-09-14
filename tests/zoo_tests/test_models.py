#!/usr/bin/env python3`
import glob
import os
import shutil

from tests import get_tests_data_path, get_tests_output_path, run_cli
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager


def test_run_all_models():
    """Check if all the models are downloadable and tts models run correctly."""
    print(" > Run synthesizer with all the models.")
    download_dir = get_user_data_dir("tts")
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    manager = ModelManager(output_prefix=get_tests_output_path())
    model_names = manager.list_models()
    for model_name in model_names:
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
                    f'--text "This is an example." --out_path "{output_path}" --speaker_idx "{speaker_id}" --language_idx "{language_id}" '
                )
            else:
                # single-speaker model
                run_cli(f"tts --model_name  {model_name} " f'--text "This is an example." --out_path "{output_path}"')
            # remove downloaded models
            shutil.rmtree(download_dir)
        else:
            # only download the model
            manager.download_model(model_name)
        print(f" | > OK: {model_name}")

    folders = glob.glob(os.path.join(manager.output_prefix, "*"))
    assert len(folders) == len(model_names)
    shutil.rmtree(manager.output_prefix)


def test_voice_conversion():
    print(" > Run voice conversion inference using YourTTS model.")
    model_name = "tts_models/multilingual/multi-dataset/your_tts"
    language_id = "en"
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    reference_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0032.wav")
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    run_cli(
        f"tts --model_name  {model_name}"
        f" --out_path {output_path} --speaker_wav {speaker_wav} --reference_wav {reference_wav} --language_idx {language_id} "
    )
