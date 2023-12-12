#!/usr/bin/env python3`
import glob
import os
import shutil

import torch

from tests import get_tests_data_path, get_tests_output_path, run_cli
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.utils.generic_utils import get_user_data_dir
from TTS.utils.manage import ModelManager

MODELS_WITH_SEP_TESTS = [
    "tts_models/multilingual/multi-dataset/bark",
    "tts_models/en/multi-dataset/tortoise-v2",
    "tts_models/multilingual/multi-dataset/xtts_v1.1",
    "tts_models/multilingual/multi-dataset/xtts_v2",
]


def run_models(offset=0, step=1):
    """Check if all the models are downloadable and tts models run correctly."""
    print(" > Run synthesizer with all the models.")
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    manager = ModelManager(output_prefix=get_tests_output_path(), progress_bar=False)
    model_names = [name for name in manager.list_models() if name not in MODELS_WITH_SEP_TESTS]
    print("Model names:", model_names)
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
    """XTTS is too big to run on github actions. We need to test it locally"""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1.1 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True '
            f'--speaker_wav "{speaker_wav}" --language_idx "en"'
        )
    else:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v1.1 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False '
            f'--speaker_wav "{speaker_wav}" --language_idx "en"'
        )


def test_xtts_streaming():
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    speaker_wav_2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0002.wav")
    speaker_wav.append(speaker_wav_2)
    model_path = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v1.1")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    print("Inference...")
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chuncks.append(chunk)
    assert len(wav_chuncks) > 1


def test_xtts_v2():
    """XTTS is too big to run on github actions. We need to test it locally"""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    speaker_wav = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")
    speaker_wav_2 = os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0002.wav")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v2 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True '
            f'--speaker_wav "{speaker_wav}" "{speaker_wav_2}"  --language_idx "en"'
        )
    else:
        run_cli(
            "yes | "
            f"tts --model_name  tts_models/multilingual/multi-dataset/xtts_v2 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False '
            f'--speaker_wav "{speaker_wav}" "{speaker_wav_2}" --language_idx "en"'
        )


def test_xtts_v2_streaming():
    """Testing the new inference_stream method"""
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    speaker_wav = [os.path.join(get_tests_data_path(), "ljspeech", "wavs", "LJ001-0001.wav")]
    model_path = os.path.join(get_user_data_dir("tts"), "tts_models--multilingual--multi-dataset--xtts_v2")
    config = XttsConfig()
    config.load_json(os.path.join(model_path, "config.json"))
    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=model_path)
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    print("Computing speaker latents...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=speaker_wav)

    print("Inference...")
    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        if i == 0:
            assert chunk.shape[-1] > 5000
        wav_chuncks.append(chunk)
    assert len(wav_chuncks) > 1
    normal_len = sum([len(chunk) for chunk in wav_chuncks])

    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=1.5,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        wav_chuncks.append(chunk)
    fast_len = sum([len(chunk) for chunk in wav_chuncks])

    chunks = model.inference_stream(
        "It took me quite a long time to develop a voice and now that I have it I am not going to be silent.",
        "en",
        gpt_cond_latent,
        speaker_embedding,
        speed=0.66,
    )
    wav_chuncks = []
    for i, chunk in enumerate(chunks):
        wav_chuncks.append(chunk)
    slow_len = sum([len(chunk) for chunk in wav_chuncks])

    assert slow_len > normal_len
    assert normal_len > fast_len


def test_tortoise():
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        run_cli(
            f" tts --model_name  tts_models/en/multi-dataset/tortoise-v2 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True'
        )
    else:
        run_cli(
            f" tts --model_name  tts_models/en/multi-dataset/tortoise-v2 "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False'
        )


def test_bark():
    """Bark is too big to run on github actions. We need to test it locally"""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    use_gpu = torch.cuda.is_available()
    if use_gpu:
        run_cli(
            f" tts --model_name  tts_models/multilingual/multi-dataset/bark "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False --use_cuda True'
        )
    else:
        run_cli(
            f" tts --model_name  tts_models/multilingual/multi-dataset/bark "
            f'--text "This is an example." --out_path "{output_path}" --progress_bar False'
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
