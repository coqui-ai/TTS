import logging
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

import tensorflow as tf
from TTS.vocoder.tf.layers.melgan import ResidualStack, ReflectionPad1d


#pylint: disable=too-many-ancestors
#pylint: disable=abstract-method
class MelganGenerator(tf.keras.models.Model):
    """ Melgan Generator TF implementation dedicated for inference with no
    weight norm """
    def __init__(self,
                 in_channels=80,
                 out_channels=1,
                 proj_kernel=7,
                 base_channels=512,
                 upsample_factors=(8, 8, 2, 2),
                 res_kernel=3,
                 num_res_blocks=3):
        super(MelganGenerator, self).__init__()

        self.in_channels = in_channels

        # assert model parameters
        assert (proj_kernel -
                1) % 2 == 0, " [!] proj_kernel should be an odd number."

        # setup additional model parameters
        base_padding = (proj_kernel - 1) // 2
        act_slope = 0.2
        self.inference_padding = 2

        # initial layer
        self.initial_layer = [
            ReflectionPad1d(base_padding),
            tf.keras.layers.Conv2D(filters=base_channels,
                                   kernel_size=(proj_kernel, 1),
                                   strides=1,
                                   padding='valid',
                                   use_bias=True,
                                   name="1")
        ]
        num_layers = 3  # count number of layers for layer naming

        # upsampling layers and residual stacks
        self.upsample_layers = []
        for idx, upsample_factor in enumerate(upsample_factors):
            layer_out_channels = base_channels // (2**(idx + 1))
            layer_filter_size = upsample_factor * 2
            layer_stride = upsample_factor
            # layer_output_padding = upsample_factor % 2
            self.upsample_layers += [
                tf.keras.layers.LeakyReLU(act_slope),
                tf.keras.layers.Conv2DTranspose(
                    filters=layer_out_channels,
                    kernel_size=(layer_filter_size, 1),
                    strides=(layer_stride, 1),
                    padding='same',
                    # output_padding=layer_output_padding,
                    use_bias=True,
                    name=f'{num_layers}'),
                ResidualStack(channels=layer_out_channels,
                              num_res_blocks=num_res_blocks,
                              kernel_size=res_kernel,
                              name=f'layers.{num_layers + 1}')
            ]
            num_layers += num_res_blocks - 1

        self.upsample_layers += [tf.keras.layers.LeakyReLU(act_slope)]

        # final layer
        self.final_layers = [
            ReflectionPad1d(base_padding),
            tf.keras.layers.Conv2D(filters=out_channels,
                                   kernel_size=(proj_kernel, 1),
                                   use_bias=True,
                                   name=f'layers.{num_layers + 1}'),
            tf.keras.layers.Activation("tanh")
        ]

        # self.model_layers = tf.keras.models.Sequential(self.initial_layer + self.upsample_layers + self.final_layers, name="layers")
        self.model_layers = self.initial_layer + self.upsample_layers + self.final_layers

    @tf.function(experimental_relax_shapes=True)
    def call(self, c, training=False):
        """
        c : B x C x T
        """
        if training:
            raise NotImplementedError()
        return self.inference(c)

    def inference(self, c):
        c = tf.transpose(c, perm=[0, 2, 1])
        c = tf.expand_dims(c, 2)
        # FIXME: TF had no replicate padding as in Torch
        # c = tf.pad(c, [[0, 0], [self.inference_padding, self.inference_padding], [0, 0], [0, 0]], "REFLECT")
        o = c
        for layer in self.model_layers:
            o = layer(o)
        # o = self.model_layers(c)
        o = tf.transpose(o, perm=[0, 3, 2, 1])
        return o[:, :, 0, :]

    def build_inference(self):
        x = tf.random.uniform((1, self.in_channels, 4), dtype=tf.float32)
        self(x, training=False)

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([1, None, None], dtype=tf.float32),
        ],)
    def inference_tflite(self, c):
        c = tf.transpose(c, perm=[0, 2, 1])
        c = tf.expand_dims(c, 2)
        # FIXME: TF had no replicate padding as in Torch
        # c = tf.pad(c, [[0, 0], [self.inference_padding, self.inference_padding], [0, 0], [0, 0]], "REFLECT")
        o = c
        for layer in self.model_layers:
            o = layer(o)
        # o = self.model_layers(c)
        o = tf.transpose(o, perm=[0, 3, 2, 1])
        return o[:, :, 0, :]
