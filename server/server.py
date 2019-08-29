#!flask/bin/python
import argparse
from synthesizer import Synthesizer
from TTS.utils.generic_utils import load_config
from flask import Flask, request, render_template, send_file

parser = argparse.ArgumentParser()
parser.add_argument(
    '-c', '--config_path', type=str, help='path to config file for training')
args = parser.parse_args()

config = load_config(args.config_path)
app = Flask(__name__)
synthesizer = Synthesizer(config)

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
    app.run(debug=config.debug, host='0.0.0.0', port=config.port)
