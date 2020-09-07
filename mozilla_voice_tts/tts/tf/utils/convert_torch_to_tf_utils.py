import numpy as np
import tensorflow as tf

# NOTE: linter has a problem with the current TF release
#pylint: disable=no-value-for-parameter
#pylint: disable=unexpected-keyword-arg

def tf_create_dummy_inputs():
    """ Create dummy inputs for TF Tacotron2 model """
    batch_size = 4
    max_input_length = 32
    max_mel_length = 128
    pad = 1
    n_chars = 24
    input_ids = tf.random.uniform([batch_size, max_input_length + pad], maxval=n_chars, dtype=tf.int32)
    input_lengths = np.random.randint(0, high=max_input_length+1 + pad, size=[batch_size])
    input_lengths[-1] = max_input_length
    input_lengths = tf.convert_to_tensor(input_lengths, dtype=tf.int32)
    mel_outputs = tf.random.uniform(shape=[batch_size, max_mel_length + pad, 80])
    mel_lengths = np.random.randint(0, high=max_mel_length+1 + pad, size=[batch_size])
    mel_lengths[-1] = max_mel_length
    mel_lengths = tf.convert_to_tensor(mel_lengths, dtype=tf.int32)
    return input_ids, input_lengths, mel_outputs, mel_lengths


def compare_torch_tf(torch_tensor, tf_tensor):
    """ Compute the average absolute difference b/w torch and tf tensors """
    return abs(torch_tensor.detach().numpy() - tf_tensor.numpy()).mean()


def convert_tf_name(tf_name):
    """ Convert certain patterns in TF layer names to Torch patterns """
    tf_name_tmp = tf_name
    tf_name_tmp = tf_name_tmp.replace(':0', '')
    tf_name_tmp = tf_name_tmp.replace('/forward_lstm/lstm_cell_1/recurrent_kernel', '/weight_hh_l0')
    tf_name_tmp = tf_name_tmp.replace('/forward_lstm/lstm_cell_2/kernel', '/weight_ih_l1')
    tf_name_tmp = tf_name_tmp.replace('/recurrent_kernel', '/weight_hh')
    tf_name_tmp = tf_name_tmp.replace('/kernel', '/weight')
    tf_name_tmp = tf_name_tmp.replace('/gamma', '/weight')
    tf_name_tmp = tf_name_tmp.replace('/beta', '/bias')
    tf_name_tmp = tf_name_tmp.replace('/', '.')
    return tf_name_tmp


def transfer_weights_torch_to_tf(tf_vars, var_map_dict, state_dict):
    """ Transfer weigths from torch state_dict to TF variables """
    print(" > Passing weights from Torch to TF ...")
    for tf_var in tf_vars:
        torch_var_name = var_map_dict[tf_var.name]
        print(f' | > {tf_var.name} <-- {torch_var_name}')
        # if tuple, it is a bias variable
        if not isinstance(torch_var_name, tuple):
            torch_layer_name = '.'.join(torch_var_name.split('.')[-2:])
            torch_weight = state_dict[torch_var_name]
        if 'convolution1d/kernel' in tf_var.name or 'conv1d/kernel' in tf_var.name:
            # out_dim, in_dim, filter -> filter, in_dim, out_dim
            numpy_weight = torch_weight.permute([2, 1, 0]).detach().cpu().numpy()
        elif 'lstm_cell' in tf_var.name and 'kernel' in tf_var.name:
            numpy_weight = torch_weight.transpose(0, 1).detach().cpu().numpy()
        # if variable is for bidirectional lstm and it is a bias vector there
        # needs to be pre-defined two matching torch bias vectors
        elif '_lstm/lstm_cell_' in tf_var.name and 'bias' in tf_var.name:
            bias_vectors = [value for key, value in state_dict.items() if key in torch_var_name]
            assert len(bias_vectors) == 2
            numpy_weight = bias_vectors[0] + bias_vectors[1]
        elif 'rnn' in tf_var.name and 'kernel' in tf_var.name:
            numpy_weight = torch_weight.transpose(0, 1).detach().cpu().numpy()
        elif 'rnn' in tf_var.name and 'bias' in tf_var.name:
            bias_vectors = [value for key, value in state_dict.items() if torch_var_name[:-2] in key]
            assert len(bias_vectors) == 2
            numpy_weight = bias_vectors[0] + bias_vectors[1]
        elif 'linear_layer' in torch_layer_name and 'weight' in torch_var_name:
            numpy_weight = torch_weight.transpose(0, 1).detach().cpu().numpy()
        else:
            numpy_weight = torch_weight.detach().cpu().numpy()
        assert np.all(tf_var.shape == numpy_weight.shape), f" [!] weight shapes does not match: {tf_var.name} vs {torch_var_name} --> {tf_var.shape} vs {numpy_weight.shape}"
        tf.keras.backend.set_value(tf_var, numpy_weight)
    return tf_vars


def load_tf_vars(model_tf, tf_vars):
    for tf_var in tf_vars:
        model_tf.get_layer(tf_var.name).set_weights(tf_var)
    return model_tf
