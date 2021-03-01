#!flask/bin/python
import argparse
import os
import sys
import io
from pathlib import Path

from flask import Flask, render_template, request, send_file
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager
from TTS.utils.io import load_config
from TTS.utils.generic_utils import style_wav_uri_to_dict


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ['true', '1', 'yes']

    parser = argparse.ArgumentParser()
    parser.add_argument('--list_models', type=convert_boolean, nargs='?', const=True, default=False, help='list available pre-trained tts and vocoder models.')
    parser.add_argument('--model_name', type=str, default="tts_models/en/ljspeech/speedy-speech-wn", help='name of one of the released tts models.')
    parser.add_argument('--vocoder_name', type=str, default=None, help='name of one of the released vocoder models.')
    parser.add_argument('--tts_checkpoint', type=str, help='path to custom tts checkpoint file')
    parser.add_argument('--tts_config', type=str, help='path to custom tts config.json file')
    parser.add_argument('--tts_speakers', type=str, help='path to JSON file containing speaker ids, if speaker ids are used in the model')
    parser.add_argument('--vocoder_config', type=str, default=None, help='path to vocoder config file.')
    parser.add_argument('--vocoder_checkpoint', type=str, default=None, help='path to vocoder checkpoint file.')
    parser.add_argument('--port', type=int, default=5002, help='port to listen on.')
    parser.add_argument('--use_cuda', type=convert_boolean, default=False, help='true to use CUDA.')
    parser.add_argument('--debug', type=convert_boolean, default=False, help='true to enable Flask debug mode.')
    parser.add_argument('--show_details', type=convert_boolean, default=False, help='Generate model detail page.')
    return parser

synthesizer = None

embedded_models_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')

embedded_tts_folder = os.path.join(embedded_models_folder, 'tts')
tts_checkpoint_file = os.path.join(embedded_tts_folder, 'checkpoint.pth.tar')
tts_config_file = os.path.join(embedded_tts_folder, 'config.json')

embedded_vocoder_folder = os.path.join(embedded_models_folder, 'vocoder')
vocoder_checkpoint_file = os.path.join(embedded_vocoder_folder, 'checkpoint.pth.tar')
vocoder_config_file = os.path.join(embedded_vocoder_folder, 'config.json')

# These models are soon to be deprecated
embedded_wavernn_folder = os.path.join(embedded_models_folder, 'wavernn')
wavernn_checkpoint_file = os.path.join(embedded_wavernn_folder, 'checkpoint.pth.tar')
wavernn_config_file = os.path.join(embedded_wavernn_folder, 'config.json')

args = create_argparser().parse_args()

path = Path(__file__).parent / "../.models.json"
manager = ModelManager(path)

if args.list_models:
    manager.list_models()
    sys.exit()

# set models by the released models
if args.model_name is not None:
    tts_checkpoint_file, tts_config_file, tts_json_dict = manager.download_model(args.model_name)
    args.vocoder_name =  tts_json_dict['default_vocoder'] if args.vocoder_name is None else args.vocoder_name

if args.vocoder_name is not None:
    vocoder_checkpoint_file, vocoder_config_file, vocoder_json_dict = manager.download_model(args.vocoder_name)

# If these were not specified in the CLI args, use default values with embedded model files
if not args.tts_checkpoint and os.path.isfile(tts_checkpoint_file):
    args.tts_checkpoint = tts_checkpoint_file
if not args.tts_config and os.path.isfile(tts_config_file):
    args.tts_config = tts_config_file

if not args.vocoder_checkpoint and os.path.isfile(vocoder_checkpoint_file):
    args.vocoder_checkpoint = vocoder_checkpoint_file
if not args.vocoder_config and os.path.isfile(vocoder_config_file):
    args.vocoder_config = vocoder_config_file


synthesizer = Synthesizer(args.tts_checkpoint, args.tts_config, args.vocoder_checkpoint, args.vocoder_config, args.use_cuda)

use_speaker_embedding = synthesizer.tts_config.get("use_external_speaker_embedding_file", False)
use_gst = synthesizer.tts_config.get("use_gst", False)
app = Flask(__name__)


@app.route('/')
def index():
    return render_template(
        'index.html', 
        show_details=args.show_details, 
        use_speaker_embedding=use_speaker_embedding,
        use_gst = use_gst
        )

@app.route('/details')
def details():
    model_config = load_config(args.tts_config)
    if args.vocoder_config is not None and os.path.isfile(args.vocoder_config):
        vocoder_config = load_config(args.vocoder_config)
    else:
        vocoder_config = None

    return render_template('details.html',
                           show_details=args.show_details
                           , model_config=model_config
                           , vocoder_config=vocoder_config
                           , args=args.__dict__
                          )

@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    speaker_json_key = request.args.get('speaker', "")
    style_wav = request.args.get('style-wav', "")
    
    style_wav = style_wav_uri_to_dict(style_wav)
    print(" > Model input: {}".format(text))
    wavs = synthesizer.tts(text, speaker_json_key=speaker_json_key, style_wav=style_wav)
    out = io.BytesIO()
    synthesizer.save_wav(wavs, out)
    return send_file(out, mimetype='audio/wav')


def main():
    app.run(debug=args.debug, host='0.0.0.0', port=args.port)


if __name__ == '__main__':
    main()
