import numpy as np


def pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]),
                  mode='constant',
                  constant_values=_pad)


def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([pad_data(x, max_len) for x in inputs])


def pad_tensor(x, length):
    _pad = 0
    assert x.ndim == 2
    return np.pad(x, [[0, 0], [0, length- x.shape[1]]], mode='constant', constant_values=_pad)


def prepare_tensor(inputs):
    max_len = max((x.shape[1] for x in inputs))
    return np.stack([pad_tensor(x, max_len) for x in inputs])


def pad_stop_target(x, length):
    _pad = 1.
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_stop_target(inputs, out_steps):
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    return np.stack([pad_stop_target(x, max_len + out_steps - remainder) for x in inputs])


def pad_per_step(inputs, pad_len):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0, 0], [0, 0],
                           [0, pad_len]],
                  mode='constant', constant_values=0.0)
