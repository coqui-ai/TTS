import tensorflow as tf


class ReflectionPad1d(tf.keras.layers.Layer):
    def __init__(self, padding):
        super(ReflectionPad1d, self).__init__()
        self.padding = padding

    def call(self, x):
        return tf.pad(x, [[0, 0], [self.padding, self.padding], [0, 0], [0, 0]], "REFLECT")


class ResidualStack(tf.keras.layers.Layer):
    def __init__(self, channels, num_res_blocks, kernel_size, name):
        super(ResidualStack, self).__init__(name=name)

        assert (kernel_size - 1) % 2 == 0, " [!] kernel_size has to be odd."
        base_padding = (kernel_size - 1) // 2

        self.blocks = []
        num_layers = 2
        for idx in range(num_res_blocks):
            layer_kernel_size = kernel_size
            layer_dilation = layer_kernel_size**idx
            layer_padding = base_padding * layer_dilation
            block = [
                tf.keras.layers.LeakyReLU(0.2),
                ReflectionPad1d(layer_padding),
                tf.keras.layers.Conv2D(filters=channels,
                                       kernel_size=(kernel_size, 1),
                                       dilation_rate=(layer_dilation, 1),
                                       use_bias=True,
                                       padding='valid',
                                       name=f'blocks.{idx}.{num_layers}'),
                tf.keras.layers.LeakyReLU(0.2),
                tf.keras.layers.Conv2D(filters=channels,
                                       kernel_size=(1, 1),
                                       use_bias=True,
                                       name=f'blocks.{idx}.{num_layers + 2}')
            ]
            self.blocks.append(block)
        self.shortcuts = [
            tf.keras.layers.Conv2D(channels,
                                   kernel_size=1,
                                   use_bias=True,
                                   name=f'shortcuts.{i}')
            for i in range(num_res_blocks)
        ]

    def call(self, x):
        for block, shortcut in zip(self.blocks, self.shortcuts):
            res = shortcut(x)
            for layer in block:
                x = layer(x)
            x += res
        return x
