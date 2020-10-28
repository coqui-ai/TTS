#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
# pylint: disable=redefined-outer-name, unused-argument
import os
import string
import time

import torch

from TTS.tts.utils.generic_utils import setup_model, is_tacotron
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.vocoder.utils.generic_utils import setup_generator


def tts(model, vocoder_model, text, CONFIG, use_cuda, ap, use_gl, speaker_fileid, speaker_embedding=None, gst_style=None):
    t_1 = time.time()
    waveform, _, _, mel_postnet_spec, _, _ = synthesis(model, text, CONFIG, use_cuda, ap, speaker_fileid, gst_style, False, CONFIG.enable_eos_bos_chars, use_gl, speaker_embedding=speaker_embedding)
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
        '--speaker_fileid',
        type=str,
        help="if CONFIG.use_external_speaker_embedding_file is true, name of speaker embedding reference file present in speakers.json, else target speaker_fileid if the model is multi-speaker.",
        default=None)
    parser.add_argument(
        '--gst_style',
        help="Wav path file for GST stylereference.",
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

    speaker_embedding = None
    speaker_embedding_dim = None
    num_speakers = 0

    # load speakers
    if args.speakers_json != '':
        speaker_mapping = json.load(open(args.speakers_json, 'r'))
        num_speakers = len(speaker_mapping)
        if C.use_external_speaker_embedding_file:
            if args.speaker_fileid is not None:
                speaker_embedding = speaker_mapping[args.speaker_fileid]['embedding']
            else: # if speaker_fileid is not specificated use the first sample in speakers.json
                speaker_embedding = speaker_mapping[list(speaker_mapping.keys())[0]]['embedding']
            speaker_embedding_dim = len(speaker_embedding)

    # load the model
    num_chars = len(phonemes) if C.use_phonemes else len(symbols)
    model = setup_model(num_chars, num_speakers, C, speaker_embedding_dim)
    cp = torch.load(args.model_path, map_location=torch.device('cpu'))
    model.load_state_dict(cp['model'])
    model.eval()
    if args.use_cuda:
        model.cuda()
    if is_tacotron(C):
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

    if not C.use_external_speaker_embedding_file:
        if args.speaker_fileid.isdigit():
            args.speaker_fileid = int(args.speaker_fileid)
        else:
            args.speaker_fileid = None
    else:
        args.speaker_fileid = None

    if args.gst_style is None:
        if is_tacotron(C):
            gst_style = C.gst['gst_style_input']
        else:
            gst_style = None
    else:
        # check if gst_style string is a dict, if is dict convert  else use string
        try:
            gst_style = json.loads(args.gst_style)
            if max(map(int, gst_style.keys())) >= C.gst['gst_style_tokens']:
                raise RuntimeError("The highest value of the gst_style dictionary key must be less than the number of GST Tokens, \n Highest dictionary key value: {} \n Number of GST tokens: {}".format(max(map(int, gst_style.keys())), C.gst['gst_style_tokens']))
        except ValueError:
            gst_style = args.gst_style

    wav = tts(model, vocoder_model, args.text, C, args.use_cuda, ap, use_griffin_lim, args.speaker_fileid, speaker_embedding=speaker_embedding, gst_style=gst_style)

    # save the results
    file_name = args.text.replace(" ", "_")
    file_name = file_name.translate(
        str.maketrans('', '', string.punctuation.replace('_', ''))) + '.wav'
    out_path = os.path.join(args.out_path, file_name)
    print(" > Saving output to {}".format(out_path))
    ap.save_wav(wav, out_path)
