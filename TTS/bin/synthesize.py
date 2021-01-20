#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import sys
import string
import time
from argparse import RawTextHelpFormatter
# pylint: disable=redefined-outer-name, unused-argument
from pathlib import Path

import numpy as np
import torch
from TTS.tts.utils.generic_utils import is_tacotron, setup_model
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.tts.utils.io import load_checkpoint
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.utils.manage import ModelManager
from TTS.vocoder.utils.generic_utils import setup_generator, interpolate_vocoder_input


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def load_tts_model(model_path, config_path, use_cuda, speakers_json=None, speaker_idx=None):
    global phonemes
    global symbols

    # load the config
    model_config = load_config(config_path)

    # load the audio processor
    ap = AudioProcessor(**model_config.audio)

    # if the vocabulary was passed, replace the default
    if 'characters' in model_config.keys():
        symbols, phonemes = make_symbols(**model_config.characters)

    # load speakers
    speaker_embedding = None
    speaker_embedding_dim = None
    num_speakers = 0
    if speakers_json is not None:
        speaker_mapping = json.load(open(speakers_json, 'r'))
        num_speakers = len(speaker_mapping)
        if model_config.use_external_speaker_embedding_file:
            if speaker_idx is not None:
                speaker_embedding = speaker_mapping[speaker_idx]['embedding']
            else: # if speaker_idx is not specificated use the first sample in speakers.json
                speaker_embedding = speaker_mapping[list(speaker_mapping.keys())[0]]['embedding']
            speaker_embedding_dim = len(speaker_embedding)

    # load tts model
    num_chars = len(phonemes) if model_config.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, model_config, speaker_embedding_dim)
    model.load_checkpoint(model_config, model_path, eval=True)
    if use_cuda:
        model.cuda()
    return model, model_config, ap, speaker_embedding


def load_vocoder_model(model_path, config_path, use_cuda):
    vocoder_config = load_config(vocoder_config_path)
    vocoder_ap = AudioProcessor(**vocoder_config['audio'])
    vocoder_model = setup_generator(vocoder_config)
    vocoder_model.load_checkpoint(vocoder_config, model_path, eval=True)
    if use_cuda:
        vocoder_model.cuda()
    return vocoder_model, vocoder_config, vocoder_ap


def tts(model,
        vocoder_model,
        text,
        model_config,
        vocoder_config,
        use_cuda,
        ap,
        vocoder_ap,
        use_gl,
        speaker_fileid,
        speaker_embedding=None,
        gst_style=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(
        model,
        text,
        model_config,
        use_cuda,
        ap,
        speaker_fileid,
        gst_style,
        False,
        model_config.enable_eos_bos_chars,
        use_gl,
        speaker_embedding=speaker_embedding)

    # grab spectrogram (thx to the nice guys at mozilla discourse for codesnippet)
    if args.save_spectogram:
        spec_file_name = args.text.replace(" ", "_")[0:10]
        spec_file_name = spec_file_name.translate(
            str.maketrans('', '', string.punctuation.replace('_', ''))) + '.npy'
        spec_file_name = os.path.join(args.out_path, spec_file_name)
        spectrogram = mel_postnet_spec.T
        spectrogram = spectrogram[0]
        np.save(spec_file_name, spectrogram)
        print(" > Saving raw spectogram to " + spec_file_name)
    # convert linear spectrogram to melspectrogram for tacotron
    if model_config.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T)
    # run vocoder_model
    if not use_gl:
        # denormalize tts output based on tts audio config
        mel_postnet_spec = ap._denormalize(mel_postnet_spec.T).T
        device_type = "cuda" if use_cuda else "cpu"
        # renormalize spectrogram based on vocoder config
        vocoder_input = vocoder_ap._normalize(mel_postnet_spec.T)
        # compute scale factor for possible sample rate mismatch
        scale_factor = [1,  vocoder_config['audio']['sample_rate'] / ap.sample_rate]
        if scale_factor[1] != 1:
            print(" > interpolating tts model output.")
            vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
        else:
            vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)
        # run vocoder model
        # [1, T, C]
        waveform = vocoder_model.inference(vocoder_input.to(device_type))
    if use_cuda and not use_gl:
        waveform = waveform.cpu()
    if not use_gl:
        waveform = waveform.numpy()
    waveform = waveform.squeeze()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    return waveform


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='''Synthesize speech on command line.\n\n'''

    '''You can either use your trained model or choose a model from the provided list.\n'''

    '''
Example runs:

    # list provided models
    ./TTS/bin/synthesize.py --list_models

    # run a model from the list
    ./TTS/bin/synthesize.py --text "Text for TTS" --model_name "<language>/<dataset>/<model_name>" --vocoder_name "<language>/<dataset>/<model_name>" --output_path

    # run your own TTS model (Using Griffin-Lim Vocoder)
    ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/model.pth.tar --config_path path/to/config.json --out_path output/path/speech.wav

    # run your own TTS and Vocoder models
    ./TTS/bin/synthesize.py --text "Text for TTS" --model_path path/to/config.json --config_path path/to/model.pth.tar --out_path output/path/speech.wav
        --vocoder_path path/to/vocoder.pth.tar --vocoder_config_path path/to/vocoder_config.json

''',
        formatter_class=RawTextHelpFormatter)

    parser.add_argument(
        '--list_models',
        type=str2bool,
        nargs='?',
        const=True,
        default=False,
        help='list available pre-trained tts and vocoder models.'
        )
    parser.add_argument(
        '--text',
        type=str,
        default=None,
        help='Text to generate speech.'
        )

    # Args for running pre-trained TTS models.
    parser.add_argument(
        '--model_name',
        type=str,
        default=None,
        help=
        'Name of one of the pre-trained tts models in format <language>/<dataset>/<model_name>'
    )
    parser.add_argument(
        '--vocoder_name',
        type=str,
        default=None,
        help=
        'Name of one of the pre-trained  vocoder models in format <language>/<dataset>/<model_name>'
    )

    # Args for running custom models
    parser.add_argument(
        '--config_path',
        default=None,
        type=str,
        help='Path to model config file.'
        )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help='Path to model file.',
    )
    parser.add_argument(
        '--out_path',
        type=str,
        default=Path(__file__).resolve().parent,
        help='Path to save final wav file. Wav file will be named as the given text.',
    )
    parser.add_argument(
        '--use_cuda',
        type=bool,
        help='Run model on CUDA.',
        default=False
        )
    parser.add_argument(
        '--vocoder_path',
        type=str,
        help=
        'Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
        default=None,
    )
    parser.add_argument(
        '--vocoder_config_path',
        type=str,
        help='Path to vocoder model config file.',
        default=None)

    # args for multi-speaker synthesis
    parser.add_argument(
        '--speakers_json',
        type=str,
        help="JSON file for multi-speaker model.",
        default=None)
    parser.add_argument(
        '--speaker_idx',
        type=str,
        help="if the tts model is trained with x-vectors, then speaker_idx is a file present in speakers.json else speaker_idx is the speaker id corresponding to a speaker in the speaker embedding layer.",
        default=None)
    parser.add_argument(
        '--gst_style',
        help="Wav path file for GST stylereference.",
        default=None)

    # aux args
    parser.add_argument(
        '--save_spectogram',
        type=bool,
        help="If true save raw spectogram for further (vocoder) processing in out_path.",
        default=False)

    args = parser.parse_args()

    # load model manager
    path = Path(__file__).parent / "../../.models.json"
    manager = ModelManager(path)

    model_path = None
    vocoder_path = None
    model = None
    vocoder_model = None
    vocoder_config = None
    vocoder_ap = None

    # CASE1: list pre-trained TTS models
    if args.list_models:
        manager.list_models()
        sys.exit()

    # CASE2: load pre-trained models
    if args.model_name is not None:
        model_path, config_path = manager.download_model(args.model_name)

    if args.vocoder_name is not None:
        vocoder_path, vocoder_config_path = manager.download_model(args.vocoder_name)

    # CASE3: load custome models
    if args.model_path is not None:
        model_path = args.model_path
        config_path = args.config_path

    if args.vocoder_path is not None:
        vocoder_path = args.vocoder_path
        vocoder_config_path = args.vocoder_config_path

    # RUN THE SYNTHESIS
    # load models
    model, model_config, ap, speaker_embedding = load_tts_model(model_path, config_path, args.use_cuda, args.speaker_idx)
    if vocoder_path is not None:
        vocoder_model, vocoder_config, vocoder_ap = load_vocoder_model(vocoder_path, vocoder_config_path, use_cuda=args.use_cuda)

    use_griffin_lim = vocoder_path is None
    print(" > Text: {}".format(args.text))

    # handle multi-speaker setting
    if not model_config.use_external_speaker_embedding_file and args.speaker_idx is not None:
        if args.speaker_idx.isdigit():
            args.speaker_idx = int(args.speaker_idx)
        else:
            args.speaker_idx = None
    else:
        args.speaker_idx = None

    if args.gst_style is None:
        if 'gst' in model_config.keys() and model_config.gst['gst_style_input'] is not None:
            gst_style = model_config.gst['gst_style_input']
        else:
            gst_style = None
    else:
        # check if gst_style string is a dict, if is dict convert  else use string
        try:
            gst_style = json.loads(args.gst_style)
            if max(map(int, gst_style.keys())) >= model_config.gst['gst_style_tokens']:
                raise RuntimeError("The highest value of the gst_style dictionary key must be less than the number of GST Tokens, \n Highest dictionary key value: {} \n Number of GST tokens: {}".format(max(map(int, gst_style.keys())), model_config.gst['gst_style_tokens']))
        except ValueError:
            gst_style = args.gst_style

    # kick it
    wav = tts(model,
              vocoder_model,
              args.text,
              model_config,
              vocoder_config,
              args.use_cuda,
              ap,
              vocoder_ap,
              use_griffin_lim,
              args.speaker_idx,
              speaker_embedding=speaker_embedding,
              gst_style=gst_style)

    # save the results
    file_name = args.text.replace(" ", "_")[0:20]
    file_name = file_name.translate(
        str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    out_path = os.path.join(args.out_path, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
