import argparse
from difflib import SequenceMatcher
import os

import numpy as np
import tensorflow as tf
import torch

from TTS.utils.io import load_config
from TTS.vocoder.tf.utils.convert_torch_to_tf_utils import (
    compare_torch_tf, convert_tf_name, transfer_weights_torch_to_tf)
from TTS.vocoder.tf.utils.generic_utils import \
    setup_generator as setup_tf_generator
from TTS.vocoder.tf.utils.io import save_checkpoint
from TTS.vocoder.utils.generic_utils import setup_generator

# prevent GPU use
os.environ['CUDA_VISIBLE_DEVICES'] = ''

# define args
parser = argparse.ArgumentParser()
parser.add_argument('--torch_model_path',
                    type=str,
                    help='Path to target torch model to be converted to TF.')
parser.add_argument('--config_path',
                    type=str,
                    help='Path to config file of torch model.')
parser.add_argument(
    '--output_path',
    type=str,
    help='path to output file including file name to save TF model.')
args = parser.parse_args()

# load model config
config_path = args.config_path
c = load_config(config_path)
num_speakers = 0

# init torch model
model = setup_generator(c)
checkpoint = torch.load(args.torch_model_path,
                        map_location=torch.device('cpu'))
state_dict = checkpoint['model']
model.load_state_dict(state_dict)
model.remove_weight_norm()
state_dict = model.state_dict()

# init tf model
model_tf = setup_tf_generator(c)

common_sufix = '/.ATTRIBUTES/VARIABLE_VALUE'
# get tf_model graph by passing an input
# B x D x T
dummy_input = tf.random.uniform((7, 80, 64), dtype=tf.float32)
mel_pred = model_tf(dummy_input, training=False)

# get tf variables
tf_vars = model_tf.weights

# match variable names with fuzzy logic
torch_var_names = list(state_dict.keys())
tf_var_names = [we.name for we in model_tf.weights]
var_map = []
for tf_name in tf_var_names:
    # skip re-mapped layer names
    if tf_name in [name[0] for name in var_map]:
        continue
    tf_name_edited = convert_tf_name(tf_name)
    ratios = [
        SequenceMatcher(None, torch_name, tf_name_edited).ratio()
        for torch_name in torch_var_names
    ]
    max_idx = np.argmax(ratios)
    matching_name = torch_var_names[max_idx]
    del torch_var_names[max_idx]
    var_map.append((tf_name, matching_name))

# pass weights
tf_vars = transfer_weights_torch_to_tf(tf_vars, dict(var_map), state_dict)

# Compare TF and TORCH models
# check embedding outputs
model.eval()
dummy_input_torch = torch.ones((1, 80, 10))
dummy_input_tf = tf.convert_to_tensor(dummy_input_torch.numpy())
dummy_input_tf = tf.transpose(dummy_input_tf, perm=[0, 2, 1])
dummy_input_tf = tf.expand_dims(dummy_input_tf, 2)

out_torch = model.layers[0](dummy_input_torch)
out_tf = model_tf.model_layers[0](dummy_input_tf)
out_tf_ = tf.transpose(out_tf, perm=[0, 3, 2, 1])[:, :, 0, :]

assert compare_torch_tf(out_torch, out_tf_) < 1e-5

for i in range(1, len(model.layers)):
    print(f"{i} -> {model.layers[i]} vs {model_tf.model_layers[i]}")
    out_torch = model.layers[i](out_torch)
    out_tf = model_tf.model_layers[i](out_tf)
    out_tf_ = tf.transpose(out_tf, perm=[0, 3, 2, 1])[:, :, 0, :]
    diff = compare_torch_tf(out_torch, out_tf_)
    assert diff < 1e-5, diff

torch.manual_seed(0)
dummy_input_torch = torch.rand((1, 80, 100))
dummy_input_tf = tf.convert_to_tensor(dummy_input_torch.numpy())
model.inference_padding = 0
model_tf.inference_padding = 0
output_torch = model.inference(dummy_input_torch)
output_tf = model_tf(dummy_input_tf, training=False)
assert compare_torch_tf(output_torch, output_tf) < 1e-5, compare_torch_tf(
    output_torch, output_tf)

# save tf model
save_checkpoint(model_tf, checkpoint['step'], checkpoint['epoch'],
                args.output_path)
print(' > Model conversion is successfully completed :).')
