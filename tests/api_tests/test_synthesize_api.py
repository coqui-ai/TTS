import os

from tests import get_tests_output_path, run_cli


def test_synthesize():
    """Test synthesize.py with diffent arguments."""
    output_path = os.path.join(get_tests_output_path(), "output.wav")

    # 🐸 Coqui studio model
    run_cli(
        'tts --model_name "coqui_studio/en/Torcull Diarmuid/coqui_studio" '
        '--text "This is it" '
        f'--out_path "{output_path}"'
    )
