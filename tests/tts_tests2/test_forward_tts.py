import torch as T

from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.utils.helpers import sequence_mask

# pylint: disable=unused-variable


def expand_encoder_outputs_test():
    model = ForwardTTS(ForwardTTSArgs(num_chars=10))

    inputs = T.rand(2, 5, 57)
    durations = T.randint(1, 4, (2, 57))

    x_mask = T.ones(2, 1, 57)
    y_mask = T.ones(2, 1, durations.sum(1).max())

    expanded, _ = model.expand_encoder_outputs(inputs, durations, x_mask, y_mask)

    for b in range(durations.shape[0]):
        index = 0
        for idx, dur in enumerate(durations[b]):
            diff = (
                expanded[b, :, index : index + dur.item()]
                - inputs[b, :, idx].repeat(dur.item()).view(expanded[b, :, index : index + dur.item()].shape)
            ).sum()
            assert abs(diff) < 1e-6, diff
            index += dur


def model_input_output_test():
    """Assert the output shapes of the model in different modes"""

    # VANILLA MODEL
    model = ForwardTTS(ForwardTTSArgs(num_chars=10, use_pitch=False, use_aligner=False))

    x = T.randint(0, 10, (2, 21))
    x_lengths = T.randint(10, 22, (2,))
    x_lengths[-1] = 21
    x_mask = sequence_mask(x_lengths).unsqueeze(1).long()
    durations = T.randint(1, 4, (2, 21))
    durations = durations * x_mask.squeeze(1)
    y_lengths = durations.sum(1)
    y_mask = sequence_mask(y_lengths).unsqueeze(1).long()

    outputs = model.forward(x, x_lengths, y_lengths, dr=durations)

    assert outputs["model_outputs"].shape == (2, durations.sum(1).max(), 80)
    assert outputs["durations_log"].shape == (2, 21)
    assert outputs["durations"].shape == (2, 21)
    assert outputs["alignments"].shape == (2, durations.sum(1).max(), 21)
    assert (outputs["x_mask"] - x_mask).sum() == 0.0
    assert (outputs["y_mask"] - y_mask).sum() == 0.0

    assert outputs["alignment_soft"] is None
    assert outputs["alignment_mas"] is None
    assert outputs["alignment_logprob"] is None
    assert outputs["o_alignment_dur"] is None
    assert outputs["pitch_avg"] is None
    assert outputs["pitch_avg_gt"] is None

    # USE PITCH
    model = ForwardTTS(ForwardTTSArgs(num_chars=10, use_pitch=True, use_aligner=False))

    x = T.randint(0, 10, (2, 21))
    x_lengths = T.randint(10, 22, (2,))
    x_lengths[-1] = 21
    x_mask = sequence_mask(x_lengths).unsqueeze(1).long()
    durations = T.randint(1, 4, (2, 21))
    durations = durations * x_mask.squeeze(1)
    y_lengths = durations.sum(1)
    y_mask = sequence_mask(y_lengths).unsqueeze(1).long()
    pitch = T.rand(2, 1, y_lengths.max())

    outputs = model.forward(x, x_lengths, y_lengths, dr=durations, pitch=pitch)

    assert outputs["model_outputs"].shape == (2, durations.sum(1).max(), 80)
    assert outputs["durations_log"].shape == (2, 21)
    assert outputs["durations"].shape == (2, 21)
    assert outputs["alignments"].shape == (2, durations.sum(1).max(), 21)
    assert (outputs["x_mask"] - x_mask).sum() == 0.0
    assert (outputs["y_mask"] - y_mask).sum() == 0.0
    assert outputs["pitch_avg"].shape == (2, 1, 21)
    assert outputs["pitch_avg_gt"].shape == (2, 1, 21)

    assert outputs["alignment_soft"] is None
    assert outputs["alignment_mas"] is None
    assert outputs["alignment_logprob"] is None
    assert outputs["o_alignment_dur"] is None

    # USE ALIGNER NETWORK
    model = ForwardTTS(ForwardTTSArgs(num_chars=10, use_pitch=False, use_aligner=True))

    x = T.randint(0, 10, (2, 21))
    x_lengths = T.randint(10, 22, (2,))
    x_lengths[-1] = 21
    x_mask = sequence_mask(x_lengths).unsqueeze(1).long()
    durations = T.randint(1, 4, (2, 21))
    durations = durations * x_mask.squeeze(1)
    y_lengths = durations.sum(1)
    y_mask = sequence_mask(y_lengths).unsqueeze(1).long()
    y = T.rand(2, y_lengths.max(), 80)

    outputs = model.forward(x, x_lengths, y_lengths, dr=durations, y=y)

    assert outputs["model_outputs"].shape == (2, durations.sum(1).max(), 80)
    assert outputs["durations_log"].shape == (2, 21)
    assert outputs["durations"].shape == (2, 21)
    assert outputs["alignments"].shape == (2, durations.sum(1).max(), 21)
    assert (outputs["x_mask"] - x_mask).sum() == 0.0
    assert (outputs["y_mask"] - y_mask).sum() == 0.0
    assert outputs["alignment_soft"].shape == (2, durations.sum(1).max(), 21)
    assert outputs["alignment_mas"].shape == (2, durations.sum(1).max(), 21)
    assert outputs["alignment_logprob"].shape == (2, 1, durations.sum(1).max(), 21)
    assert outputs["o_alignment_dur"].shape == (2, 21)

    assert outputs["pitch_avg"] is None
    assert outputs["pitch_avg_gt"] is None

    # USE ALIGNER NETWORK AND PITCH
    model = ForwardTTS(ForwardTTSArgs(num_chars=10, use_pitch=True, use_aligner=True))

    x = T.randint(0, 10, (2, 21))
    x_lengths = T.randint(10, 22, (2,))
    x_lengths[-1] = 21
    x_mask = sequence_mask(x_lengths).unsqueeze(1).long()
    durations = T.randint(1, 4, (2, 21))
    durations = durations * x_mask.squeeze(1)
    y_lengths = durations.sum(1)
    y_mask = sequence_mask(y_lengths).unsqueeze(1).long()
    y = T.rand(2, y_lengths.max(), 80)
    pitch = T.rand(2, 1, y_lengths.max())

    outputs = model.forward(x, x_lengths, y_lengths, dr=durations, pitch=pitch, y=y)

    assert outputs["model_outputs"].shape == (2, durations.sum(1).max(), 80)
    assert outputs["durations_log"].shape == (2, 21)
    assert outputs["durations"].shape == (2, 21)
    assert outputs["alignments"].shape == (2, durations.sum(1).max(), 21)
    assert (outputs["x_mask"] - x_mask).sum() == 0.0
    assert (outputs["y_mask"] - y_mask).sum() == 0.0
    assert outputs["alignment_soft"].shape == (2, durations.sum(1).max(), 21)
    assert outputs["alignment_mas"].shape == (2, durations.sum(1).max(), 21)
    assert outputs["alignment_logprob"].shape == (2, 1, durations.sum(1).max(), 21)
    assert outputs["o_alignment_dur"].shape == (2, 21)
    assert outputs["pitch_avg"].shape == (2, 1, 21)
    assert outputs["pitch_avg_gt"].shape == (2, 1, 21)
