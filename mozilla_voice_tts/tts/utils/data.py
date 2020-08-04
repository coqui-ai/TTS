import numpy as np


def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])


def _pad_tensor(x, length):
    _pad = 0.
    assert x.ndim == 2
    x = np.pad(
        x, [[0, 0], [0, length - x.shape[1]]],
        mode='constant',
        constant_values=_pad)
    return x


def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[1] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])


def _pad_stop_target(x, length):
    _pad = 0.
    assert x.ndim == 1
    return np.pad(
        x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_stop_target(inputs, out_steps):
    """ Pad row vectors with 1. """
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])


def pad_per_step(inputs, pad_len):
    return np.pad(
        inputs, [[0, 0], [0, 0], [0, pad_len]],
        mode='constant',
        constant_values=0.0)


# pylint: disable=attribute-defined-outside-init
class StandardScaler():

    def set_stats(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def reset_stats(self):
        delattr(self, 'mean_')
        delattr(self, 'scale_')

    def transform(self, X):
        X = np.asarray(X)
        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X = np.asarray(X)
        X *= self.scale_
        X += self.mean_
        return X
