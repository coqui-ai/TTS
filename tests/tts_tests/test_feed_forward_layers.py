import torch

from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.utils.helpers import sequence_mask

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_encoder():
    input_dummy = torch.rand(8, 14, 37).to(device)
    input_lengths = torch.randint(31, 37, (8,)).long().to(device)
    input_lengths[-1] = 37
    input_mask = torch.unsqueeze(sequence_mask(input_lengths, input_dummy.size(2)), 1).to(device)
    # relative positional transformer encoder
    layer = Encoder(
        out_channels=11,
        in_hidden_channels=14,
        encoder_type="relative_position_transformer",
        encoder_params={
            "hidden_channels_ffn": 768,
            "num_heads": 2,
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 6,
            "rel_attn_window_size": 4,
            "input_length": None,
        },
    ).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]
    # residual conv bn encoder
    layer = Encoder(
        out_channels=11,
        in_hidden_channels=14,
        encoder_type="residual_conv_bn",
        encoder_params={"kernel_size": 4, "dilations": 4 * [1, 2, 4] + [1], "num_conv_blocks": 2, "num_res_blocks": 13},
    ).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]
    # FFTransformer encoder
    layer = Encoder(
        out_channels=14,
        in_hidden_channels=14,
        encoder_type="fftransformer",
        encoder_params={"hidden_channels_ffn": 31, "num_heads": 2, "num_layers": 2, "dropout_p": 0.1},
    ).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 14, 37]


def test_decoder():
    input_dummy = torch.rand(8, 128, 37).to(device)
    input_lengths = torch.randint(31, 37, (8,)).long().to(device)
    input_lengths[-1] = 37

    input_mask = torch.unsqueeze(sequence_mask(input_lengths, input_dummy.size(2)), 1).to(device)
    # residual bn conv decoder
    layer = Decoder(out_channels=11, in_hidden_channels=128).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]
    # transformer decoder
    layer = Decoder(
        out_channels=11,
        in_hidden_channels=128,
        decoder_type="relative_position_transformer",
        decoder_params={
            "hidden_channels_ffn": 128,
            "num_heads": 2,
            "kernel_size": 3,
            "dropout_p": 0.1,
            "num_layers": 8,
            "rel_attn_window_size": 4,
            "input_length": None,
        },
    ).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]
    # wavenet decoder
    layer = Decoder(
        out_channels=11,
        in_hidden_channels=128,
        decoder_type="wavenet",
        decoder_params={
            "num_blocks": 12,
            "hidden_channels": 192,
            "kernel_size": 5,
            "dilation_rate": 1,
            "num_layers": 4,
            "dropout_p": 0.05,
        },
    ).to(device)
    output = layer(input_dummy, input_mask)
    # FFTransformer decoder
    layer = Decoder(
        out_channels=11,
        in_hidden_channels=128,
        decoder_type="fftransformer",
        decoder_params={
            "hidden_channels_ffn": 31,
            "num_heads": 2,
            "dropout_p": 0.1,
            "num_layers": 2,
        },
    ).to(device)
    output = layer(input_dummy, input_mask)
    assert list(output.shape) == [8, 11, 37]
