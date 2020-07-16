import numpy as np
import tensorflow as tf

from scipy import signal as sig


class PQMF(tf.keras.layers.Layer):
    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()
        # define filter coefficient
        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi /
                                             (2 * N)) * (np.arange(taps + 1) -
                                                         ((taps - 1) / 2))
            phase = (-1)**k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)

            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        # [N, 1, taps + 1] == [filter_width, in_channels, out_channels]
        self.H = np.transpose(H[:, None, :], (2, 1, 0)).astype('float32')
        self.G = np.transpose(G[None, :, :], (2, 1, 0)).astype('float32')

        # filter for downsampling & upsampling
        updown_filter = np.zeros((N, N, N), dtype=np.float32)
        for k in range(N):
            updown_filter[0, k, k] = 1.0
        self.updown_filter = updown_filter.astype(np.float32)

    def analysis(self, x):
        """
        x : B x 1 x T
        """
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]], constant_values=0.0)
        x = tf.nn.conv1d(x, self.H, stride=1, padding='VALID')
        x = tf.nn.conv1d(x,
                         self.updown_filter,
                         stride=self.N,
                         padding='VALID')
        x = tf.transpose(x, perm=[0, 2, 1])
        return x

    def synthesis(self, x):
        """
        x : B x D x T
        """
        x = tf.transpose(x, perm=[0, 2, 1])
        x = tf.nn.conv1d_transpose(
            x,
            self.updown_filter * self.N,
            strides=self.N,
            output_shape=(tf.shape(x)[0], tf.shape(x)[1] * self.N,
                          self.N))
        x = tf.pad(x, [[0, 0], [self.taps // 2, self.taps // 2], [0, 0]], constant_values=0.0)
        x = tf.nn.conv1d(x, self.G, stride=1, padding="VALID")
        x = tf.transpose(x, perm=[0, 2, 1])
        return x
