import datetime
import pickle
import tensorflow as tf


def save_checkpoint(model, current_step, epoch, output_path, **kwargs):
    """ Save TF Vocoder model """
    state = {
        'model': model.weights,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
    }
    state.update(kwargs)
    pickle.dump(state, open(output_path, 'wb'))


def load_checkpoint(model, checkpoint_path):
    """ Load TF Vocoder model """
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    chkp_var_dict = {var.name: var.numpy() for var in checkpoint['model']}
    tf_vars = model.weights
    for tf_var in tf_vars:
        layer_name = tf_var.name
        chkp_var_value = chkp_var_dict[layer_name]
        tf.keras.backend.set_value(tf_var, chkp_var_value)
    return model
