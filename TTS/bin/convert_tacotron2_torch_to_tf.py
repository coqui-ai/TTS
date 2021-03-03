import argparse
from difflib import SequenceMatcher
import os
import sys
from pprint import pprint

import numpy as np
import tensorflow as tf
import torch
from TTS.tts.tf.models.tacotron2 import Tacotron2
from TTS.tts.tf.utils.convert_torch_to_tf_utils import (
    compare_torch_tf, convert_tf_name, transfer_weights_torch_to_tf)
from TTS.tts.tf.utils.generic_utils import save_checkpoint
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.text.symbols import phonemes, symbols
from TTS.utils.io import load_config

sys.path.append('/home/erogol/Projects')
os.environ['CUDA_VISIBLE_DEVICES'] = ''


parser = argparse.ArgumentParser()
parser.add_argument('--torch_model_path',
                    type=str,
                    help='Path to target torch model to be converted to TF.')
parser.add_argument('--config_path',
                    type=str,
                    help='Path to config file of torch model.')
parser.add_argument('--output_path',
                    type=str,
                    help='path to output file including file name to save TF model.')
args = parser.parse_args()

# load model config
config_path = args.config_path
c = load_config(config_path)
num_speakers = 0

# init torch model
num_chars = len(phonemes) if c.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speakers, c)
checkpoint = torch.load(args.torch_model_path,
                        map_location=torch.device('cpu'))
state_dict = checkpoint['model']
model.load_state_dict(state_dict)

# init tf model
model_tf = Tacotron2(num_chars=num_chars,
                     num_speakers=num_speakers,
                     r=model.decoder.r,
                     postnet_output_dim=c.audio['num_mels'],
                     decoder_output_dim=c.audio['num_mels'],
                     attn_type=c.attention_type,
                     attn_win=c.windowing,
                     attn_norm=c.attention_norm,
                     prenet_type=c.prenet_type,
                     prenet_dropout=c.prenet_dropout,
                     forward_attn=c.use_forward_attn,
                     trans_agent=c.transition_agent,
                     forward_attn_mask=c.forward_attn_mask,
                     location_attn=c.location_attn,
                     attn_K=c.attention_heads,
                     separate_stopnet=c.separate_stopnet,
                     bidirectional_decoder=c.bidirectional_decoder)

# set initial layer mapping - these are not captured by the below heuristic approach
# TODO: set layer names so that we can remove these manual matching
common_sufix = '/.ATTRIBUTES/VARIABLE_VALUE'
var_map = [
    ('embedding/embeddings:0', 'embedding.weight'),
    ('encoder/lstm/forward_lstm/lstm_cell_1/kernel:0',
     'encoder.lstm.weight_ih_l0'),
    ('encoder/lstm/forward_lstm/lstm_cell_1/recurrent_kernel:0',
     'encoder.lstm.weight_hh_l0'),
    ('encoder/lstm/backward_lstm/lstm_cell_2/kernel:0',
     'encoder.lstm.weight_ih_l0_reverse'),
    ('encoder/lstm/backward_lstm/lstm_cell_2/recurrent_kernel:0',
     'encoder.lstm.weight_hh_l0_reverse'),
    ('encoder/lstm/forward_lstm/lstm_cell_1/bias:0',
     ('encoder.lstm.bias_ih_l0', 'encoder.lstm.bias_hh_l0')),
    ('encoder/lstm/backward_lstm/lstm_cell_2/bias:0',
     ('encoder.lstm.bias_ih_l0_reverse', 'encoder.lstm.bias_hh_l0_reverse')),
    ('attention/v/kernel:0', 'decoder.attention.v.linear_layer.weight'),
    ('decoder/linear_projection/kernel:0',
     'decoder.linear_projection.linear_layer.weight'),
    ('decoder/stopnet/kernel:0', 'decoder.stopnet.1.linear_layer.weight')
]

# %%
# get tf_model graph
model_tf.build_inference()

# get tf variables
tf_vars = model_tf.weights

# match variable names with fuzzy logic
torch_var_names = list(state_dict.keys())
tf_var_names = [we.name for we in model_tf.weights]
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

pprint(var_map)
pprint(torch_var_names)

# pass weights
tf_vars = transfer_weights_torch_to_tf(tf_vars, dict(var_map), state_dict)

# Compare TF and TORCH models
# %%
# check embedding outputs
model.eval()
input_ids = torch.randint(0, 24, (1, 128)).long()

o_t = model.embedding(input_ids)
o_tf = model_tf.embedding(input_ids.detach().numpy())
assert abs(o_t.detach().numpy() -
           o_tf.numpy()).sum() < 1e-5, abs(o_t.detach().numpy() -
                                           o_tf.numpy()).sum()

# compare encoder outputs
oo_en = model.encoder.inference(o_t.transpose(1, 2))
ooo_en = model_tf.encoder(o_t.detach().numpy(), training=False)
assert compare_torch_tf(oo_en, ooo_en) < 1e-5

#pylint: disable=redefined-builtin
# compare decoder.attention_rnn
inp = torch.rand([1, 768])
inp_tf = inp.numpy()
model.decoder._init_states(oo_en, mask=None)  #pylint: disable=protected-access
output, cell_state = model.decoder.attention_rnn(inp)
states = model_tf.decoder.build_decoder_initial_states(1, 512, 128)
output_tf, memory_state = model_tf.decoder.attention_rnn(inp_tf,
                                                         states[2],
                                                         training=False)
assert compare_torch_tf(output, output_tf).mean() < 1e-5

query = output
inputs = torch.rand([1, 128, 512])
query_tf = query.detach().numpy()
inputs_tf = inputs.numpy()

# compare decoder.attention
model.decoder.attention.init_states(inputs)
processes_inputs = model.decoder.attention.preprocess_inputs(inputs)
loc_attn, proc_query = model.decoder.attention.get_location_attention(
    query, processes_inputs)
context = model.decoder.attention(query, inputs, processes_inputs, None)

attention_states = model_tf.decoder.build_decoder_initial_states(1, 512, 128)[-1]
model_tf.decoder.attention.process_values(tf.convert_to_tensor(inputs_tf))
loc_attn_tf, proc_query_tf = model_tf.decoder.attention.get_loc_attn(query_tf, attention_states)
context_tf, attention, attention_states = model_tf.decoder.attention(query_tf, attention_states, training=False)

assert compare_torch_tf(loc_attn, loc_attn_tf).mean() < 1e-5
assert compare_torch_tf(proc_query, proc_query_tf).mean() < 1e-5
assert compare_torch_tf(context, context_tf) < 1e-5

# compare decoder.decoder_rnn
input = torch.rand([1, 1536])
input_tf = input.numpy()
model.decoder._init_states(oo_en, mask=None)  #pylint: disable=protected-access
output, cell_state = model.decoder.decoder_rnn(
    input, [model.decoder.decoder_hidden, model.decoder.decoder_cell])
states = model_tf.decoder.build_decoder_initial_states(1, 512, 128)
output_tf, memory_state = model_tf.decoder.decoder_rnn(input_tf,
                                                       states[3],
                                                       training=False)
assert abs(input - input_tf).mean() < 1e-5
assert compare_torch_tf(output, output_tf).mean() < 1e-5

# compare decoder.linear_projection
input = torch.rand([1, 1536])
input_tf = input.numpy()
output = model.decoder.linear_projection(input)
output_tf = model_tf.decoder.linear_projection(input_tf, training=False)
assert compare_torch_tf(output, output_tf) < 1e-5

# compare decoder outputs
model.decoder.max_decoder_steps = 100
model_tf.decoder.set_max_decoder_steps(100)
output, align, stop = model.decoder.inference(oo_en)
states = model_tf.decoder.build_decoder_initial_states(1, 512, 128)
output_tf, align_tf, stop_tf = model_tf.decoder(ooo_en, states, training=False)
assert compare_torch_tf(output.transpose(1, 2), output_tf) < 1e-4

# compare the whole model output
outputs_torch = model.inference(input_ids)
outputs_tf = model_tf(tf.convert_to_tensor(input_ids.numpy()))
print(abs(outputs_torch[0].numpy()[:, 0] - outputs_tf[0].numpy()[:, 0]).mean())
assert compare_torch_tf(outputs_torch[2][:, 50, :],
                        outputs_tf[2][:, 50, :]) < 1e-5
assert compare_torch_tf(outputs_torch[0], outputs_tf[0]) < 1e-4

# %%
# save tf model
save_checkpoint(model_tf, None, checkpoint['step'], checkpoint['epoch'],
                checkpoint['r'], args.output_path)
print(' > Model conversion is successfully completed :).')
