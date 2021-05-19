import numpy as np
import tensorflow as tf

from TTS.vocoder.tf.models.melgan_generator import MelganGenerator


def test_melgan_generator():
    hop_length = 256
    model = MelganGenerator()
    # pylint: disable=no-value-for-parameter
    dummy_input = tf.random.uniform((4, 80, 64))
    output = model(dummy_input, training=False)
    assert np.all(output.shape == (4, 1, 64 * hop_length)), output.shape
