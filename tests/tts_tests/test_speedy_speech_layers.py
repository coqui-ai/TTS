import torch

from TTS.tts.configs import SpeedySpeechConfig
from TTS.tts.layers.feed_forward.duration_predictor import DurationPredictor
from TTS.tts.models.speedy_speech import SpeedySpeech, SpeedySpeechArgs
from TTS.tts.utils.data import sequence_mask

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_duration_predictor():
    input_dummy = torch.rand(8, 128, 27).to(device)
    input_lengths = torch.randint(20, 27, (8,)).long().to(device)
    input_lengths[-1] = 27

    x_mask = torch.unsqueeze(sequence_mask(input_lengths, input_dummy.size(2)), 1).to(device)

    layer = DurationPredictor(hidden_channels=128).to(device)

    output = layer(input_dummy, x_mask)
    assert list(output.shape) == [8, 1, 27]


def test_speedy_speech():
    num_chars = 7
    B = 8
    T_en = 37
    T_de = 74

    x_dummy = torch.randint(0, 7, (B, T_en)).long().to(device)
    x_lengths = torch.randint(31, T_en, (B,)).long().to(device)
    x_lengths[-1] = T_en

    # set durations. max total duration should be equal to T_de
    durations = torch.randint(1, 4, (B, T_en))
    durations = durations * (T_de / durations.sum(1)).unsqueeze(1)
    durations = durations.to(torch.long).to(device)
    max_dur = durations.sum(1).max()
    durations[:, 0] += T_de - max_dur if T_de > max_dur else 0

    y_lengths = durations.sum(1)

    config = SpeedySpeechConfig(model_args=SpeedySpeechArgs(num_chars=num_chars, out_channels=80, hidden_channels=128))
    model = SpeedySpeech(config)
    if use_cuda:
        model.cuda()

    # forward pass
    outputs = model(x_dummy, x_lengths, y_lengths, durations)
    o_de = outputs["model_outputs"]
    attn = outputs["alignments"]
    o_dr = outputs["durations_log"]

    assert list(o_de.shape) == [B, T_de, 80], f"{list(o_de.shape)}"
    assert list(attn.shape) == [B, T_de, T_en]
    assert list(o_dr.shape) == [B, T_en]

    # with speaker embedding
    config = SpeedySpeechConfig(
        model_args=SpeedySpeechArgs(
            num_chars=num_chars, out_channels=80, hidden_channels=128, num_speakers=80, d_vector_dim=256
        )
    )
    model = SpeedySpeech(config).to(device)
    model.forward(
        x_dummy, x_lengths, y_lengths, durations, aux_input={"d_vectors": torch.randint(0, 10, (B,)).to(device)}
    )
    o_de = outputs["model_outputs"]
    attn = outputs["alignments"]
    o_dr = outputs["durations_log"]

    assert list(o_de.shape) == [B, T_de, 80], f"{list(o_de.shape)}"
    assert list(attn.shape) == [B, T_de, T_en]
    assert list(o_dr.shape) == [B, T_en]

    # with speaker external embedding
    config = SpeedySpeechConfig(
        model_args=SpeedySpeechArgs(
            num_chars=num_chars,
            out_channels=80,
            hidden_channels=128,
            num_speakers=10,
            use_d_vector=True,
            d_vector_dim=256,
        )
    )
    model = SpeedySpeech(config).to(device)
    model.forward(x_dummy, x_lengths, y_lengths, durations, aux_input={"d_vectors": torch.rand((B, 256)).to(device)})
    o_de = outputs["model_outputs"]
    attn = outputs["alignments"]
    o_dr = outputs["durations_log"]

    assert list(o_de.shape) == [B, T_de, 80], f"{list(o_de.shape)}"
    assert list(attn.shape) == [B, T_de, T_en]
    assert list(o_dr.shape) == [B, T_en]
