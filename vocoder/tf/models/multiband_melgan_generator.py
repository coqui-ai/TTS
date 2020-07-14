import tensorflow as tf

from TTS.vocoder.tf.models.melgan_generator import MelganGenerator
from TTS.vocoder.tf.layers.pqmf import PQMF

#pylint: disable=too-many-ancestors
#pylint: disable=abstract-method
class MultibandMelganGenerator(MelganGenerator):
    def __init__(self,
                 in_channels=80,
                 out_channels=4,
                 proj_kernel=7,
                 base_channels=384,
                 upsample_factors=(2, 8, 2, 2),
                 res_kernel=3,
                 num_res_blocks=3):
        super(MultibandMelganGenerator,
              self).__init__(in_channels=in_channels,
                             out_channels=out_channels,
                             proj_kernel=proj_kernel,
                             base_channels=base_channels,
                             upsample_factors=upsample_factors,
                             res_kernel=res_kernel,
                             num_res_blocks=num_res_blocks)
        self.pqmf_layer = PQMF(N=4, taps=62, cutoff=0.15, beta=9.0)

    def pqmf_analysis(self, x):
        return self.pqmf_layer.analysis(x)

    def pqmf_synthesis(self, x):
        return self.pqmf_layer.synthesis(x)

    def inference(self, c):
        c = tf.transpose(c, perm=[0, 2, 1])
        c = tf.expand_dims(c, 2)
        # FIXME: TF had no replicate padding as in Torch
        # c = tf.pad(c, [[0, 0], [self.inference_padding, self.inference_padding], [0, 0], [0, 0]], "REFLECT")
        o = c
        for layer in self.model_layers:
            o = layer(o)
        o = tf.transpose(o, perm=[0, 3, 2, 1])
        o = self.pqmf_layer.synthesis(o[:, :, 0, :])
        return o

    @tf.function(
        experimental_relax_shapes=True,
        input_signature=[
            tf.TensorSpec([1, 80, None], dtype=tf.float32),
        ],)
    def inference_tflite(self, c):
        c = tf.transpose(c, perm=[0, 2, 1])
        c = tf.expand_dims(c, 2)
        # FIXME: TF had no replicate padding as in Torch
        # c = tf.pad(c, [[0, 0], [self.inference_padding, self.inference_padding], [0, 0], [0, 0]], "REFLECT")
        o = c
        for layer in self.model_layers:
            o = layer(o)
        o = tf.transpose(o, perm=[0, 3, 2, 1])
        o = self.pqmf_layer.synthesis(o[:, :, 0, :])
        return o
