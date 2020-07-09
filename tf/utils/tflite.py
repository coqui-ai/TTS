import tensorflow as tf


def convert_tacotron2_to_tflite(model):
    tacotron2_concrete_function = model.inference_tflite.get_concrete_function()
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [tacotron2_concrete_function]
    )
    converter.experimental_new_converter = True
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                        tf.lite.OpsSet.SELECT_TF_OPS]
    tflite_model = converter.convert()
    return tflite_model


def load_tflite_model(tflite_path):
    tflite_model = tf.lite.Interpreter(model_path=tflite_path)
    tflite_model.allocate_tensors()
    return tflite_model