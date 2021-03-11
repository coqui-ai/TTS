import torch

from TTS.tts.layers.speedy_speech.encoder import Encoder
from TTS.tts.layers.speedy_speech.decoder import Decoder
from TTS.tts.layers.speedy_speech.duration_predictor import DurationPredictor
from TTS.tts.utils.generic_utils import sequence_mask
from TTS.tts.models.speedy_speech import SpeedySpeech


use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_encoder():
    input_dummy = torch.rand(8, 14, 37).to(device)
    input_lengths = torch.randint(31, 37, (8, )).long().to(device)
    input_lengths[-1] = 37
    input_mask = torch.unsqueeze(
        sequence_mask(input_lengths, input_dummy.size(2)), 1).to(device)

    # residual bn conv encoder
    layer = Encoder(out_channels=11,
                    in_hidden_channels=14,
                    encoder_type='residual_conv_bn').to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]

    # transformer encoder
    layer = Encoder(out_channels=11,
                    in_hidden_channels=14,
                    encoder_type='transformer',
                    encoder_params={
                        'hidden_channels_ffn': 768,
                        'num_heads': 2,
                        "kernel_size": 3,
                        "dropout_p": 0.1,
                        "num_layers": 6,
                        "rel_attn_window_size": 4,
                        "input_length": None
                    }).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]


def test_decoder():
    input_dummy = torch.rand(8, 128, 37).to(device)
    input_lengths = torch.randint(31, 37, (8, )).long().to(device)
    input_lengths[-1] = 37

    input_mask = torch.unsqueeze(
        sequence_mask(input_lengths, input_dummy.size(2)), 1).to(device)

    # residual bn conv decoder
    layer = Decoder(out_channels=11, in_hidden_channels=128).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]

    # transformer decoder
    layer = Decoder(out_channels=11,
                    in_hidden_channels=128,
                    decoder_type='transformer',
                    decoder_params={
                        'hidden_channels_ffn': 128,
                        'num_heads': 2,
                        "kernel_size": 3,
                        "dropout_p": 0.1,
                        "num_layers": 8,
                        "rel_attn_window_size": 4,
                        "input_length": None
                    }).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]


    # wavenet decoder
    layer = Decoder(out_channels=11,
                    in_hidden_channels=128,
                    decoder_type='wavenet',
                    decoder_params={
                        "num_blocks": 12,
                        "hidden_channels": 192,
                        "kernel_size": 5,
                        "dilation_rate": 1,
                        "num_layers": 4,
                        "dropout_p": 0.05
                    }).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]



def test_duration_predictor():
    input_dummy = torch.rand(8, 128, 27).to(device)
    input_lengths = torch.randint(20, 27, (8, )).long().to(device)
    input_lengths[-1] = 27

    x_mask = torch.unsqueeze(sequence_mask(input_lengths, input_dummy.size(2)),
                             1).to(device)

    layer = DurationPredictor(hidden_channels=128).to(device)

    output = layer(input_dummy, x_mask)
    assert list(output.shape) == [8, 1, 27]


def test_speedy_speech():
    num_chars = 7
    B = 8
    T_en = 37
    T_de = 74

    x_dummy = torch.randint(0, 7, (B, T_en)).long().to(device)
    x_lengths = torch.randint(31, T_en, (B, )).long().to(device)
    x_lengths[-1] = T_en

    # set durations. max total duration should be equal to T_de
    durations = torch.randint(1, 4, (B, T_en))
    durations = durations * (T_de / durations.sum(1)).unsqueeze(1)
    durations = durations.to(torch.long).to(device)
    max_dur = durations.sum(1).max()
    durations[:, 0] += T_de - max_dur if T_de > max_dur else 0

    y_lengths = durations.sum(1)

    model = SpeedySpeech(num_chars, out_channels=80, hidden_channels=128)
    if use_cuda:
        model.cuda()

    # forward pass
    o_de, o_dr, attn = model(x_dummy, x_lengths, y_lengths, durations)

    assert list(o_de.shape) == [B, 80, T_de], f"{list(o_de.shape)}"
    assert list(attn.shape) == [B, T_de, T_en]
    assert list(o_dr.shape) == [B, T_en]

    # with speaker embedding
    model = SpeedySpeech(num_chars,
                         out_channels=80,
                         hidden_channels=128,
                         num_speakers=10,
                         c_in_channels=256).to(device)
    model.forward(x_dummy,
                  x_lengths,
                  y_lengths,
                  durations,
                  g=torch.randint(0, 10, (B,)).to(device))

    assert list(o_de.shape) == [B, 80, T_de], f"{list(o_de.shape)}"
    assert list(attn.shape) == [B, T_de, T_en]
    assert list(o_dr.shape) == [B, T_en]


    # with speaker external embedding
    model = SpeedySpeech(num_chars,
                         out_channels=80,
                         hidden_channels=128,
                         num_speakers=10,
                         external_c=True,
                         c_in_channels=256).to(device)
    model.forward(x_dummy,
                  x_lengths,
                  y_lengths,
                  durations,
                  g=torch.rand((B, 256)).to(device))

    assert list(o_de.shape) == [B, 80, T_de], f"{list(o_de.shape)}"
    assert list(attn.shape) == [B, T_de, T_en]
    assert list(o_dr.shape) == [B, T_en]
