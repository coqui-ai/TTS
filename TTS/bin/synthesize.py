#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

from TTS.api import TTS
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
    description = """Synthesize speech on command line.

You can either use your trained model or choose a model from the provided list.

If you don't specify any models, then it uses LJSpeech based English model.

## Example Runs

### Single Speaker Models

- List provided models:

    ```
    $ tts --list_models
    ```

- Query info for model info by idx:

    ```
    $ tts --model_info_by_idx "<model_type>/<model_query_idx>"
    ```

- Query info for model info by full name:

    ```
    $ tts --model_info_by_name "<model_type>/<language>/<dataset>/<model_name>"
    ```

- Run TTS with default models:

    ```
    $ tts --text "Text for TTS"
    ```

- Run a TTS model with its default vocoder model:

    ```
    $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>
    ```

- Run with specific TTS and vocoder models from the list:

    ```
    $ tts --text "Text for TTS" --model_name "<model_type>/<language>/<dataset>/<model_name>" --vocoder_name "<model_type>/<language>/<dataset>/<model_name>" --output_path
    ```

- Run your own TTS model (Using Griffin-Lim Vocoder):

    ```
    $ tts --text "Text for TTS" --model_path path/to/model.pth --config_path path/to/config.json --out_path output/path/speech.wav
    ```

- Run your own TTS and Vocoder models:
    ```
    $ tts --text "Text for TTS" --model_path path/to/config.json --config_path path/to/model.pth --out_path output/path/speech.wav
        --vocoder_path path/to/vocoder.pth --vocoder_config_path path/to/vocoder_config.json
    ```

### Multi-speaker Models

- List the available speakers and choose as <speaker_id> among them:

    ```
    $ tts --model_name "<language>/<dataset>/<model_name>"  --list_speaker_idxs
    ```

- Run the multi-speaker TTS model with the target speaker ID:

    ```
    $ tts --text "Text for TTS." --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>"  --speaker_idx <speaker_id>
    ```

- Run your own multi-speaker TTS model:

    ```
    $ tts --text "Text for TTS" --out_path output/path/speech.wav --model_path path/to/config.json --config_path path/to/model.pth --speakers_file_path path/to/speaker.json --speaker_idx <speaker_id>
    ```

### Voice Conversion Models

    ```
    $ tts --out_path output/path/speech.wav --model_name "<language>/<dataset>/<model_name>" --source_wav <path/to/speaker/wav> --target_wav <path/to/reference/wav>
    ```
    """
    # We remove Markdown code formatting programmatically here to allow us to copy-and-paste from main README to keep
    # documentation in sync more easily.
    parser = argparse.ArgumentParser(
        description=description.replace("    ```\n", ""),
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument(
        "--list_models",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="list available pre-trained TTS and vocoder models.",
    )

    parser.add_argument(
        "--model_info_by_idx",
        type=str,
        default=None,
        help="model info using query format: <model_type>/<model_query_idx>",
    )

    parser.add_argument(
        "--model_info_by_name",
        type=str,
        default=None,
        help="model info using query format: <model_type>/<language>/<dataset>/<model_name>",
    )

    parser.add_argument("--text", type=str, default=None, help="Text to generate speech.")

    # Args for running pre-trained TTS models.
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/ljspeech/tacotron2-DDC",
        help="Name of one of the pre-trained TTS models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--vocoder_name",
        type=str,
        default=None,
        help="Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>",
    )

    # Args for running custom models
    parser.add_argument("--config_path", default=None, type=str, help="Path to model config file.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model file.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="tts_output.wav",
        help="Output wav file path.",
    )
    parser.add_argument("--use_cuda", type=bool, help="Run model on CUDA.", default=False)
    parser.add_argument("--device", type=str, help="Device to run model on.", default="cpu")
    parser.add_argument(
        "--vocoder_path",
        type=str,
        help="Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).",
        default=None,
    )
    parser.add_argument("--vocoder_config_path", type=str, help="Path to vocoder model config file.", default=None)
    parser.add_argument(
        "--encoder_path",
        type=str,
        help="Path to speaker encoder model file.",
        default=None,
    )
    parser.add_argument("--encoder_config_path", type=str, help="Path to speaker encoder config file.", default=None)

    # args for coqui studio
    parser.add_argument(
        "--cs_model",
        type=str,
        help="Name of the 🐸Coqui Studio model. Available models are `XTTS`, `XTTS-multilingual`, `V1`.",
    )
    parser.add_argument(
        "--emotion",
        type=str,
        help="Emotion to condition the model with. Only available for 🐸Coqui Studio `V1` model.",
        default=None,
    )
    parser.add_argument(
        "--language",
        type=str,
        help="Language to condition the model with. Only available for 🐸Coqui Studio `XTTS-multilingual` model.",
        default=None,
    )

    # args for multi-speaker synthesis
    parser.add_argument("--speakers_file_path", type=str, help="JSON file for multi-speaker model.", default=None)
    parser.add_argument("--language_ids_file_path", type=str, help="JSON file for multi-lingual model.", default=None)
    parser.add_argument(
        "--speaker_idx",
        type=str,
        help="Target speaker ID for a multi-speaker TTS model.",
        default=None,
    )
    parser.add_argument(
        "--language_idx",
        type=str,
        help="Target language ID for a multi-lingual TTS model.",
        default=None,
    )
    parser.add_argument(
        "--speaker_wav",
        nargs="+",
        help="wav file(s) to condition a multi-speaker TTS model with a Speaker Encoder. You can give multiple file paths. The d_vectors is computed as their average.",
        default=None,
    )
    parser.add_argument("--gst_style", help="Wav path file for GST style reference.", default=None)
    parser.add_argument(
        "--capacitron_style_wav", type=str, help="Wav path file for Capacitron prosody reference.", default=None
    )
    parser.add_argument("--capacitron_style_text", type=str, help="Transcription of the reference.", default=None)
    parser.add_argument(
        "--list_speaker_idxs",
        help="List available speaker ids for the defined multi-speaker model.",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
    )
    parser.add_argument(
        "--list_language_idxs",
        help="List available language ids for the defined multi-lingual model.",
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
    parser.add_argument(
        "--reference_wav",
        type=str,
        help="Reference wav file to convert in the voice of the speaker_idx or speaker_wav",
        default=None,
    )
    parser.add_argument(
        "--reference_speaker_idx",
        type=str,
        help="speaker ID of the reference_wav speaker (If not provided the embedding will be computed using the Speaker Encoder).",
        default=None,
    )
    parser.add_argument(
        "--progress_bar",
        type=str2bool,
        help="If true shows a progress bar for the model download. Defaults to True",
        default=True,
    )

    # voice conversion args
    parser.add_argument(
        "--source_wav",
        type=str,
        default=None,
        help="Original audio file to convert in the voice of the target_wav",
    )
    parser.add_argument(
        "--target_wav",
        type=str,
        default=None,
        help="Target audio file to convert in the voice of the source_wav",
    )

    parser.add_argument(
        "--voice_dir",
        type=str,
        default=None,
        help="Voice dir for tortoise model",
    )

    args = parser.parse_args()

    # print the description if either text or list_models is not set
    check_args = [
        args.text,
        args.list_models,
        args.list_speaker_idxs,
        args.list_language_idxs,
        args.reference_wav,
        args.model_info_by_idx,
        args.model_info_by_name,
        args.source_wav,
        args.target_wav,
    ]
    if not any(check_args):
        parser.parse_args(["-h"])

    # load model manager
    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path, progress_bar=args.progress_bar)
    api = TTS()

    tts_path = None
    tts_config_path = None
    speakers_file_path = None
    language_ids_file_path = None
    vocoder_path = None
    vocoder_config_path = None
    encoder_path = None
    encoder_config_path = None
    vc_path = None
    vc_config_path = None
    model_dir = None

    # CASE1 #list : list pre-trained TTS models
    if args.list_models:
        manager.add_cs_api_models(api.list_models())
        manager.list_models()
        sys.exit()

    # CASE2 #info : model info for pre-trained TTS models
    if args.model_info_by_idx:
        model_query = args.model_info_by_idx
        manager.model_info_by_idx(model_query)
        sys.exit()

    if args.model_info_by_name:
        model_query_full_name = args.model_info_by_name
        manager.model_info_by_full_name(model_query_full_name)
        sys.exit()

    # CASE3: TTS with coqui studio models
    if "coqui_studio" in args.model_name:
        print(" > Using 🐸Coqui Studio model: ", args.model_name)
        api = TTS(model_name=args.model_name, cs_api_model=args.cs_model)
        api.tts_to_file(text=args.text, emotion=args.emotion, file_path=args.out_path, language=args.language)
        print(" > Saving output to ", args.out_path)
        return

    # CASE4: load pre-trained model paths
    if args.model_name is not None and not args.model_path:
        model_path, config_path, model_item = manager.download_model(args.model_name)
        # tts model
        if model_item["model_type"] == "tts_models":
            tts_path = model_path
            tts_config_path = config_path
            if "default_vocoder" in model_item:
                args.vocoder_name = model_item["default_vocoder"] if args.vocoder_name is None else args.vocoder_name

        # voice conversion model
        if model_item["model_type"] == "voice_conversion_models":
            vc_path = model_path
            vc_config_path = config_path

        # tts model with multiple files to be loaded from the directory path
        if model_item.get("author", None) == "fairseq" or isinstance(model_item["model_url"], list):
            model_dir = model_path
            tts_path = None
            tts_config_path = None
            args.vocoder_name = None

    # load vocoder
    if args.vocoder_name is not None and not args.vocoder_path:
        vocoder_path, vocoder_config_path, _ = manager.download_model(args.vocoder_name)

    # CASE5: set custom model paths
    if args.model_path is not None:
        tts_path = args.model_path
        tts_config_path = args.config_path
        speakers_file_path = args.speakers_file_path
        language_ids_file_path = args.language_ids_file_path

    if args.vocoder_path is not None:
        vocoder_path = args.vocoder_path
        vocoder_config_path = args.vocoder_config_path

    if args.encoder_path is not None:
        encoder_path = args.encoder_path
        encoder_config_path = args.encoder_config_path
    
    device = args.device
    if args.use_cuda:
        device = "cuda"

    # load models
    synthesizer = Synthesizer(
        tts_path,
        tts_config_path,
        speakers_file_path,
        language_ids_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        vc_path,
        vc_config_path,
        model_dir,
        args.voice_dir,
    ).to(device)

    # query speaker ids of a multi-speaker model.
    if args.list_speaker_idxs:
        print(
            " > Available speaker ids: (Set --speaker_idx flag to one of these values to use the multi-speaker model."
        )
        print(synthesizer.tts_model.speaker_manager.name_to_id)
        return

    # query langauge ids of a multi-lingual model.
    if args.list_language_idxs:
        print(
            " > Available language ids: (Set --language_idx flag to one of these values to use the multi-lingual model."
        )
        print(synthesizer.tts_model.language_manager.name_to_id)
        return

    # check the arguments against a multi-speaker model.
    if synthesizer.tts_speakers_file and (not args.speaker_idx and not args.speaker_wav):
        print(
            " [!] Looks like you use a multi-speaker model. Define `--speaker_idx` to "
            "select the target speaker. You can list the available speakers for this model by `--list_speaker_idxs`."
        )
        return

    # RUN THE SYNTHESIS
    if args.text:
        print(" > Text: {}".format(args.text))

    # kick it
    if tts_path is not None:
        wav = synthesizer.tts(
            args.text,
            speaker_name=args.speaker_idx,
            language_name=args.language_idx,
            speaker_wav=args.speaker_wav,
            reference_wav=args.reference_wav,
            style_wav=args.capacitron_style_wav,
            style_text=args.capacitron_style_text,
            reference_speaker_name=args.reference_speaker_idx,
        )
    elif vc_path is not None:
        wav = synthesizer.voice_conversion(
            source_wav=args.source_wav,
            target_wav=args.target_wav,
        )
    elif model_dir is not None:
        wav = synthesizer.tts(args.text, speaker_name=args.speaker_idx)

    # save the results
    print(" > Saving output to {}".format(args.out_path))
    synthesizer.save_wav(wav, args.out_path)


if __name__ == "__main__":
    main()
