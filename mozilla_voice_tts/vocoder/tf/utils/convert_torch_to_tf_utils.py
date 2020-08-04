import numpy as np
import tensorflow as tf


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
        if 'kernel' in tf_var.name:
            torch_weight = state_dict[torch_var_name]
            numpy_weight = torch_weight.permute([2, 1, 0]).numpy()[:, None, :, :]
        if 'bias' in tf_var.name:
            torch_weight = state_dict[torch_var_name]
            numpy_weight = torch_weight
        assert np.all(tf_var.shape == numpy_weight.shape), f" [!] weight shapes does not match: {tf_var.name} vs {torch_var_name} --> {tf_var.shape} vs {numpy_weight.shape}"
        tf.keras.backend.set_value(tf_var, numpy_weight)
    return tf_vars


def load_tf_vars(model_tf, tf_vars):
    for tf_var in tf_vars:
        model_tf.get_layer(tf_var.name).set_weights(tf_var)
    return model_tf
