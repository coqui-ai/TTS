"""Find all the unique characters in a dataset"""
import argparse
import os
from argparse import RawTextHelpFormatter

from TTS.tts.datasets import _get_preprocessor_by_name


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Find all the unique characters or phonemes in a dataset.\n\n"""
        """Target dataset must be defined in TTS.tts.datasets.formatters\n\n"""
        """
    Example runs:

    python TTS/bin/find_unique_chars.py --dataset ljspeech --meta_file /path/to/LJSpeech/metadata.csv
    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--dataset", type=str, default="", help="One of the target dataset names in TTS.tts.datasets.formatters."
    )

    parser.add_argument("--meta_file", type=str, default=None, help="Path to the transcriptions file of the dataset.")

    args = parser.parse_args()

    preprocessor = _get_preprocessor_by_name(args.dataset)
    items = preprocessor(os.path.dirname(args.meta_file), os.path.basename(args.meta_file))
    texts = "".join(item[0] for item in items)
    chars = set(texts)
    lower_chars = filter(lambda c: c.islower(), chars)
    print(f" > Number of unique characters: {len(chars)}")
    print(f" > Unique characters: {''.join(sorted(chars))}")
    print(f" > Unique lower characters: {''.join(sorted(lower_chars))}")


if __name__ == "__main__":
    main()
