#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
# pylint: disable=redefined-outer-name, unused-argument
import os
import string
import time

import torch

from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator


def tts(model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_id):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, stop_tokens, _ = synthesis(model, text, CONFIG, use_cuda, ap, speaker_id, None, False, CONFIG.enable_eos_bos_chars, use_gl)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
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

    global symbols, phonemes

    parser = argparse.ArgumentParser()
    parser.add_argument('text', type=str, help='Text to generate speech.')
    parser.add_argument('config_path',
                        type=str,
                        help='Path to model config file.')
    parser.add_argument(
        'model_path',
        type=str,
        help='Path to model file.',
    )
    parser.add_argument(
        'out_path',
        type=str,
        help='Path to save final wav file. Wav file will be names as the text given.',
    )
    parser.add_argument('--use_cuda',
                        type=bool,
                        help='Run model on CUDA.',
                        default=False)
    parser.add_argument(
        '--vocoder_path',
        type=str,
        help=
        'Path to vocoder model file. If it is not defined, model uses GL as vocoder. Please make sure that you installed vocoder library before (WaveRNN).',
        default="",
    )
    parser.add_argument('--vocoder_config_path',
                        type=str,
                        help='Path to vocoder model config file.',
                        default="")
    parser.add_argument(
        '--batched_vocoder',
        type=bool,
        help="If True, vocoder model uses faster batch processing.",
        default=True)
    parser.add_argument('--speakers_json',
                        type=str,
                        help="JSON file for multi-speaker model.",
                        default="")
    parser.add_argument(
        '--speaker_id',
        type=int,
        help="target speaker_id if the model is multi-speaker.",
        default=None)
    args = parser.parse_args()

    # load the config
    C = load_config(args.config_path)
    C.forward_attn_mask = True

    # load the audio processor
    ap = AudioProcessor(**C.audio)

    # if the vocabulary was passed, replace the default
    if 'characters' in C.keys():
        symbols, phonemes = make_symbols(**C.characters)

    # load speakers
    if args.speakers_json != '':
        speakers = json.load(open(args.speakers_json, 'r'))
        num_speakers = len(speakers)
    else:
        num_speakers = 0

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C)
    cp = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model'])
    model.eval()
    if args.use_cuda:
        model.cuda()
    model.decoder.set_r(cp['r'])

    # load vocoder model
    if args.vocoder_path != "":
        VC = load_config(args.vocoder_config_path)
        vocoder_model = setup_generator(VC)
        vocoder_model.load_state_dict(torch.load(args.vocoder_path, map_location="cpu")["model"])
        vocoder_model.remove_weight_norm()
        if args.use_cuda:
            vocoder_model.cuda()
        vocoder_model.eval()
    else:
        vocoder_model = None
        VC = None

    # synthesize voice
    use_griffin_lim = args.vocoder_path == ""
    print(" > Text: {}".format(args.text))
    wav = tts(model, vocoder_model, args.text, C, args.use_cuda, ap, use_griffin_lim, args.speaker_id)

    # save the results
    file_name = args.text.replace(" ", "_")
    file_name = file_name.translate(
        str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    out_path = os.path.join(args.out_path, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
