#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import os
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    if v.lower() in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")


def main():
    # pylint: disable=bad-option-value
    parser = argparse.ArgumentParser(
        description="""Synthesize speech on command line.\n\n"""
        """You can either use your trained model or choose a model from the provided list.\n\n"""
        """If you don't specify any models, then it uses LJSpeech based English model.\n\n"""
        """
    # Example Runs:

    ## Single Speaker Models

    - list provided models

    ```
    $ ./TTS/bin/synthesize.py --list_models
    ```

    - run tts with default models.

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS"
    ```

    - run tts with input from stdin.

    ```
    $ echo "Text for TTS" | ./TTS/bin/synthesize.py
    ```

    - run a tts model with its default vocoder model.

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_name "<language>/<dataset>/<model_name>
    ```

    - run with specific tts and vocoder models from the list

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_name "<language>/<dataset>/<model_name>" --vocoder_name "<language>/<dataset>/<model_name>" --output_path
    ```

    - run your own TTS model (Using Griffin-Lim Vocoder)

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/model.pth.tar --config_path path/to/config.json --out_path output/path/speech.wav
    ```

    - run your own TTS and Vocoder models
    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/config.json --config_path path/to/model.pth.tar --out_path output/path/speech.wav
        --vocoder_path path/to/vocoder.pth.tar --vocoder_config_path path/to/vocoder_config.json
    ```

    ## MULTI-SPEAKER MODELS

    - list the available speakers and choose as <speaker_id> among them.

    ```
    $ ./TTS/bin/synthesize.py --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
    ```

    - run the multi-speaker TTS model with the target speaker ID.

    ```
    $ ./TTS/bin/synthesize.py --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
    ```

    - run your own multi-speaker TTS model.

    ```
    $ ./TTS/bin/synthesize.py --list_models
    $ ./TTS/bin/synthesize.py --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/config.json --config_path path/to/model.pth.tar --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
    ```

    ## CSV INPUT

    - create a WAV file for each line of a CSV file (id|text format)

    ```
    $ cat << EOF |
    s01|The birch canoe slid on the smooth planks.
    s02|Glue the sheet to the dark blue background.
    s03|It's easy to tell the depth of a well.
    s04|These days a chicken leg is a rare dish.
    s05|Rice is often served in round bowls.
    s06|The juice of lemons makes fine punch.
    s07|The box was thrown beside the parked truck.
    s08|The hogs were fed chopped corn and garbage.
    s09|Four hours of steady work faced us.
    s10|Large size in stockings is hard to sell.
    EOF
        ./TTS/bin/synthesize.py --csv --out_path wavs
    ```

    - use different speakers for a multi-speaker model (id|speaker_idx|text)

    $ cat << EOF |
    s01|speaker1|This is spoken with speaker 1.
    s02|speaker2|This is spoken with speaker 2.
    s03|This is spoken with the default speaker.
    EOF
        ./TTS/bin/synthesize.py --csv --out_path wavs --model_name "<language>/<dataset>/<model_name>" --speaker_idx <default_speaker>
    ```

    """,
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--list_models",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained tts and vocoder models.",
    )
    parser.add_argument("--text", type=str, default=None, help="Text to generate speech (use stdin if not provided).")

    # Args for running pre-trained TTS models.
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default=None,
        help="Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>",
    )

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model file.")
    parser.add_argument("--out_path", type=str, default="tts_output.wav", help="Output wav file path.")
    parser.add_argument("--use_cuda", type=bool, help="Run model on CUDA.", default=False)
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument("--encoder_path", type=str, help="Path to speaker encoder model file.", default=None)
    parser.add_argument("--encoder_config_path", type=str, help="Path to speaker encoder config file.", default=None)

    # args for multi-speaker synthesis
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument(
        "--speaker_idx", type=str, help="Target speaker ID for a multi-speaker TTS model.", default=None
    )
    parser.add_argument(
        "--speaker_wav",
        nargs="+",
        help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
        default=None,
    )
    parser.add_argument("--gst_style", help="Wav path file for GST stylereference.", default=None)
    parser.add_argument(
        "--list_speaker_idxs",
        help="List available speaker ids for the defined multi-speaker model.",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    # aux args
    parser.add_argument(
        "--save_spectogram",
        type=bool,
        help="If true save raw spectogram for further (vocoder) processing in out_path.",
        default=False,
    )
    # csv args
    parser.add_argument(
        "--csv",
        action="store_true",
        help="Input text is lines with the format id|text or id|speaker|text and out_path is a directory",
    )
    parser.add_argument(
        "--csv_delimiter", default="|", help="Delimiter used to separate id and text with --csv (default: |)"
    )

    args = parser.parse_args()

    # Read text from stdin if --text is not set
    if args.text is None and not args.list_models and not args.list_speaker_idxs:
        if os.isatty(sys.stdin.fileno()):
            print("Reading input from stdin. Hit CTRL+D to end.", file=sys.stderr)

    # load model manager
    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    model_path = None
    config_path = None
    speakers_file_path = None
    vocoder_path = None
    vocoder_config_path = None
    encoder_path = None
    encoder_config_path = None

    # CASE1: list pre-trained TTS models
    if args.list_models:
        manager.list_models()
        sys.exit()

    # CASE2: load pre-trained model paths
    if args.model_name is not None and not args.model_path:
        model_path, config_path, model_item = manager.download_model(args.model_name)
        args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

    if args.vocoder_name is not None and not args.vocoder_path:
        vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

    # CASE3: set custom model paths
    if args.model_path is not None:
        model_path = args.model_path
        config_path = args.config_path
        speakers_file_path = args.speakers_file_path

    if args.vocoder_path is not None:
        vocoder_path = args.vocoder_path
        vocoder_config_path = args.vocoder_config_path

    if args.encoder_path is not None:
        encoder_path = args.encoder_path
        encoder_config_path = args.encoder_config_path

    # load models
    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        args.use_cuda,
    )

    # query speaker ids of a multi-speaker model.
    if args.list_speaker_idxs:
        print(
            " > Available speaker ids: (Set --speaker_idx flag to one of these values to use the multi-speaker model."
        )
        print(synthesizer.tts_model.speaker_manager.speaker_ids)
        return

    # check the arguments against a multi-speaker model.
    if synthesizer.tts_speakers_file and (not args.speaker_idx and not args.speaker_wav):
        print(
            " [!] Looks like you use a multi-speaker model. Define `--speaker_idx` to "
            "select the target speaker. You can list the available speakers for this model by `--list_speaker_idxs`."
        )
        return

    # RUN THE SYNTHESIS
    if args.csv:
        # Input text is CSV with an id and text on each line.
        # out_path should be a directory instead of a name.
        # Each input line produces a WAV file at {out_path}/{id}.wav
        #
        # For multi-speaker models, an input line may contain id, speaker_idx, and text.
        # If no speaker_idx is given, --speaker_idx is used as a default.
        if os.path.splitext(args.out_path)[1] == ".wav":
            print(" [!] out_path looks like a file name, but it should be a directory for CSV input.")
            return

        os.makedirs(args.out_path, exist_ok=True)

        if args.text is not None:
            # Use --text input
            csv_input = io.StringIO(args.text)
        else:
            # Use stdin input
            csv_input = sys.stdin

        reader = csv.reader(csv_input, delimiter=args.csv_delimiter)
        for row_index, row in enumerate(reader):
            if len(row) < 2:
                # Malformed row
                print("Row %s has less than 2 columns (%s)", row_index + 1, row)
                continue

            if len(row) == 2:
                # id|text
                text_id, text = row[0], row[1]
                speaker_idx = args.speaker_idx
            elif len(row) > 2:
                # id|speaker_idx|text
                text_id, speaker_idx, text = row[0], row[1], row[2]

            print(" > Text: {}".format(text))

            # kick it
            wav = synthesizer.tts(text, speaker_idx, args.speaker_wav)

            # save the results to a {out_path}/{id}.wav
            out_path = os.path.join(args.out_path, text_id) + ".wav"
            print(" > Saving output to {}".format(out_path))
            synthesizer.save_wav(wav, out_path)
    else:
        if args.text is not None:
            # Use --text input
            text = args.text
        else:
            # Use stdin input
            text = sys.stdin.read()

        print(" > Text: {}".format(text))

        # kick it
        wav = synthesizer.tts(text, args.speaker_idx, args.speaker_wav)

        # save the results
        print(" > Saving output to {}".format(args.out_path))
        synthesizer.save_wav(wav, args.out_path)


if __name__ == "__main__":
    main()
