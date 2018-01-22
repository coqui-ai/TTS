import numpy as np


def pad_data(x, length):
    _pad = 0
    return np.pad(x, (0, length - x.shape[0]), mode='constant', constant_values=_pad)


def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([pad_data(x, max_len) for x in inputs])


def pad_per_step(inputs, outputs_per_step):
    timesteps = inputs.shape[-1]
    return np.pad(inputs, [[0, 0], [0, 0],
                           [0, outputs_per_step - (timesteps % outputs_per_step)]],
                  mode='constant', constant_values=0.0)
