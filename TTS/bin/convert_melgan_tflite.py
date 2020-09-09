# Convert Tensorflow Tacotron2 model to TF-Lite binary

import argparse

from TTS.utils.io import load_config
from TTS.vocoder.tf.utils.generic_utils import setup_generator
from TTS.vocoder.tf.utils.io import load_checkpoint
from TTS.vocoder.tf.utils.tflite import convert_melgan_to_tflite


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
model = setup_generator(CONFIG)
model.build_inference()
model = load_checkpoint(model, args.tf_model)

# create tflite model
tflite_model = convert_melgan_to_tflite(model, output_path=args.output_path)
