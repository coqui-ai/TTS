# Convert Tensorflow Tacotron2 model to TF-Lite binary

import argparse

from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.tts.tf.utils.generic_utils import setup_model
from TTS.tts.tf.utils.io import load_checkpoint
from TTS.tts.tf.utils.tflite import convert_tacotron2_to_tflite


parser = argparse.ArgumentParser()
parser.add_argument('--tf_model',
                    type=str,
                    help='Path to target torch model to be converted to TF.')
parser.add_argument('--config_path',
                    type=str,
                    help='Path to config file of torch model.')
parser.add_argument('--output_path',
                    type=str,
                    help='path to tflite output binary.')
args = parser.parse_args()

# Set constants
CONFIG = load_config(args.config_path)

# load the model
c = CONFIG
num_speakers = 0
num_chars = len(phonemes) if c.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speakers, c, enable_tflite=True)
model.build_inference()
model = load_checkpoint(model, args.tf_model)
model.decoder.set_max_decoder_steps(1000)

# create tflite model
tflite_model = convert_tacotron2_to_tflite(model, output_path=args.output_path)
