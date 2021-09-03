import unittest

import torch as T

from TTS.tts.layers.losses import L1LossMasked, SSIMLoss
from TTS.tts.layers.tacotron.tacotron import CBHG, Decoder, Encoder, Prenet
from TTS.tts.models.fast_pitch import FastPitch, FastPitchArgs, average_pitch
from TTS.tts.utils.data import sequence_mask

# pylint: disable=unused-variable


class AveragePitchTests(unittest.TestCase):
    def test_in_out(self):  # pylint: disable=no-self-use
        pitch = T.rand(1, 1, 128)

        durations = T.randint(1, 5, (1, 21))
        coeff = 128.0 / durations.sum()
        durations = T.round(durations * coeff)
        diff = 128.0 - durations.sum()
        durations[0, -1] += diff
        durations = durations.long()

        pitch_avg = average_pitch(pitch, durations)

        index = 0
        for idx, dur in enumerate(durations[0]):
            assert abs(pitch_avg[0, 0, idx] - pitch[0, 0, index : index + dur.item()].mean()) < 1e-5
            index += dur


def expand_encoder_outputs_test():
    model = FastPitch(FastPitchArgs(num_chars=10))

    inputs = T.rand(2, 5, 57)
    durations = T.randint(1, 4, (2, 57))

    x_mask = T.ones(2, 1, 57)
    y_mask = T.ones(2, 1, durations.sum(1).max())

    expanded, attn = model.expand_encoder_outputs(inputs, durations, x_mask, y_mask)

    for b in range(durations.shape[0]):
        index = 0
        for idx, dur in enumerate(durations[b]):
            diff = (
                expanded[b, :, index : index + dur.item()]
                - inputs[b, :, idx].repeat(dur.item()).view(expanded[b, :, index : index + dur.item()].shape)
            ).sum()
            assert abs(diff) < 1e-6, diff
            index += dur
