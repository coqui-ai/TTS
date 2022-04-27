import os

from tests import get_tests_output_path, run_cli


def test_synthesize():
    """Test synthesize.py with diffent arguments."""
    output_path = os.path.join(get_tests_output_path(), "output.wav")
    run_cli("tts --list_models")

    # single speaker model
    run_cli(f'tts --text "This is an example." --out_path "{output_path}"')
    run_cli(
        "tts --model_name tts_models/en/ljspeech/glow-tts " f'--text "This is an example." --out_path "{output_path}"'
    )
    run_cli(
        "tts --model_name tts_models/en/ljspeech/glow-tts  "
        "--vocoder_name vocoder_models/en/ljspeech/multiband-melgan "
        f'--text "This is an example." --out_path "{output_path}"'
    )

    # multi-speaker SC-Glow model
    # run_cli("tts --model_name tts_models/en/vctk/sc-glow-tts --list_speaker_idxs")
    # run_cli(
    #     f'tts --model_name tts_models/en/vctk/sc-glow-tts --speaker_idx "p304" '
    #     f'--text "This is an example." --out_path "{output_path}"'
    # )
