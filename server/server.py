#!flask/bin/python
import argparse
import os

from flask import Flask, request, render_template, send_file
from TTS.server.synthesizer import Synthesizer


def create_argparser():
    def convert_boolean(x):
        return x.lower() in ['true', '1', 'yes']

    parser = argparse.ArgumentParser()
    parser.add_argument('--tts_checkpoint', type=str, help='path to TTS checkpoint file')
    parser.add_argument('--tts_config', type=str, help='path to TTS config.json file')
    parser.add_argument('--tts_speakers', type=str, help='path to JSON file containing speaker ids, if speaker ids are used in the model')
    parser.add_argument('--wavernn_lib_path', type=str, default=None, help='path to WaveRNN project folder to be imported. If this is not passed, model uses Griffin-Lim for synthesis.')
    parser.add_argument('--wavernn_file', type=str, default=None, help='path to WaveRNN checkpoint file.')
    parser.add_argument('--wavernn_config', type=str, default=None, help='path to WaveRNN config file.')
    parser.add_argument('--is_wavernn_batched', type=convert_boolean, default=False, help='true to use batched WaveRNN.')
    parser.add_argument('--pwgan_lib_path', type=str, help='path to ParallelWaveGAN project folder to be imported. If this is not passed, model uses Griffin-Lim for synthesis.')
    parser.add_argument('--pwgan_file', type=str, help='path to ParallelWaveGAN checkpoint file.')
    parser.add_argument('--pwgan_config', type=str, help='path to ParallelWaveGAN config file.')
    parser.add_argument('--port', type=int, default=5002, help='port to listen on.')
    parser.add_argument('--use_cuda', type=convert_boolean, default=False, help='true to use CUDA.')
    parser.add_argument('--debug', type=convert_boolean, default=False, help='true to enable Flask debug mode.')
    return parser


synthesizer = None

embedded_model_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'model')
checkpoint_file = os.path.join(embedded_model_folder, 'checkpoint.pth.tar')
config_file = os.path.join(embedded_model_folder, 'config.json')

# Default options with embedded model files
if os.path.isfile(checkpoint_file):
    default_tts_checkpoint = checkpoint_file
else:
    default_tts_checkpoint = None

if os.path.isfile(config_file):
    default_tts_config = config_file
else:
    default_tts_config = None

args = create_argparser().parse_args()

# If these were not specified in the CLI args, use default values
if not args.tts_checkpoint:
    args.tts_checkpoint = default_tts_checkpoint
if not args.tts_config:
    args.tts_config = default_tts_config

synthesizer = Synthesizer(args)

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tts', methods=['GET'])
def tts():
    text = request.args.get('text')
    print(" > Model input: {}".format(text))
    data = synthesizer.tts(text)
    return send_file(data, mimetype='audio/wav')


if __name__ == '__main__':
    app.run(debug=args.debug, host='0.0.0.0', port=args.port)
