import pickle
import datetime
import tensorflow as tf


def save_checkpoint(model, optimizer, current_step, epoch, r, output_path, **kwargs):
    state = {
        'model': model.weights,
        'optimizer': optimizer,
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime("%B %d, %Y"),
        'r': r
    }
    state.update(kwargs)
    pickle.dump(state, open(output_path, 'wb'))


def load_checkpoint(model, checkpoint_path):
    checkpoint = pickle.load(open(checkpoint_path, 'rb'))
    chkp_var_dict = {var.name: var.numpy() for var in checkpoint['model']}
    tf_vars = model.weights
    for tf_var in tf_vars:
        layer_name = tf_var.name
        try:
            chkp_var_value = chkp_var_dict[layer_name]
        except KeyError:
            class_name = list(chkp_var_dict.keys())[0].split("/")[0]
            layer_name = f"{class_name}/{layer_name}"
            chkp_var_value = chkp_var_dict[layer_name]

        tf.keras.backend.set_value(tf_var, chkp_var_value)
    if 'r' in checkpoint.keys():
        model.decoder.set_r(checkpoint['r'])
    return model


def load_tflite_model(tflite_path):
    tflite_model = tf.lite.Interpreter(model_path=tflite_path)
    tflite_model.allocate_tensors()
    return tflite_model
