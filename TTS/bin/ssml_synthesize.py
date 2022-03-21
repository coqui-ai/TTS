import argparse
import sys
from argparse import RawTextHelpFormatter

# pylint: disable=redefined-outer-name, unused-argument
import numpy as np
from pathlib import Path
from gruut import sentences
from matplotlib.style import available

import xml
import xml.etree.ElementTree as ET
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer

def main():
    description = """Synthesize speech from an SSML file.
You can either use your trained model or choose a model from the provided list.
"""
    parser = argparse.ArgumentParser(
        description=description.replace("    ```\n", ""),
        formatter_class=RawTextHelpFormatter,
    )

    parser.add_argument("--file", type=str, default=None, help="Path to the SSML file.")
    parser.add_argument("--use_cuda", type=bool, help="Run model on CUDA.", default=False)
    parser.add_argument(
        "--model_name",
        type=str,
        default="tts_models/en/vctk/vits",
        help="Name of one of the pre-trained TTS models in format <language>/<dataset>/<model_name>",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="tts_output.wav",
        help="Output wav file path.",
    )
    args = parser.parse_args()

    path = Path(__file__).parent / "../.models.json"
    manager = ModelManager(path)

    model_path = None
    config_path = None
    speakers_file_path = None
    language_ids_file_path = None
    vocoder_path = None
    vocoder_config_path = None
    encoder_path = None
    encoder_config_path = None

    if args.file is None:
        print("Please specify the SSML file.")
        sys.exit(1)

    if args.model_name is not None:
        model_path, config_path, _ = manager.download_model(args.model_name)

    synthesizer = Synthesizer(
        model_path,
        config_path,
        speakers_file_path,
        language_ids_file_path,
        vocoder_path,
        vocoder_config_path,
        encoder_path,
        encoder_config_path,
        args.use_cuda,
    )

    with open(args.file, "r") as f:
        ssml_text = f.read()

    available_speakers = list(synthesizer.tts_model.speaker_manager.speaker_ids.keys())
    default_speaker = available_speakers[0]

    full_sent = []
    orderedSpeakers = []
    for sent in sentences(ssml_text, ssml=True, espeak=True):
        sub_sent = []
        current_speaker = None
        index = -1
        for word in sent:
            #print(sub_sent)
            if word.voice is None:
                word.voice = default_speaker
            if word.voice != current_speaker:
                current_speaker = word.voice
                sub_sent.append([''.join(word.phonemes)])
                orderedSpeakers.append(current_speaker)
                index += 1
            else:
                sub_sent[index].append(''.join(word.phonemes))
        full_sent.append(sub_sent)

    wavs = []
    for sub_sent in full_sent:
        for sent in sub_sent:
            sent = ' '.join(sent)
            print(sent)
            wavs.append(synthesizer.tts(sent, orderedSpeakers[len(wavs)], None, None, ssml=True))

    final_wav = np.array([], dtype=np.float32)
    for wav in wavs:
        for sub_wav in wav:
            final_wav = np.append(final_wav, sub_wav)

    print(" > Saving output to {}".format(args.out_path))
    synthesizer.save_wav(final_wav, args.out_path)


if __name__ == "__main__":
    main()