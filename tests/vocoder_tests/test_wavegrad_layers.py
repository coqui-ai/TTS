import torch

from TTS.vocoder.configs import WavegradConfig
from TTS.vocoder.layers.wavegrad import DBlock, FiLM, PositionalEncoding, UBlock
from TTS.vocoder.models.wavegrad import Wavegrad, WavegradArgs


def test_positional_encoding():
    layer = PositionalEncoding(50)
    inp = torch.rand(32, 50, 100)
    nl = torch.rand(32)
    o = layer(inp, nl)

    assert o.shape[0] == 32
    assert o.shape[1] == 50
    assert o.shape[2] == 100
    assert isinstance(o, torch.FloatTensor)


def test_film():
    layer = FiLM(50, 76)
    inp = torch.rand(32, 50, 100)
    nl = torch.rand(32)
    shift, scale = layer(inp, nl)

    assert shift.shape[0] == 32
    assert shift.shape[1] == 76
    assert shift.shape[2] == 100
    assert isinstance(shift, torch.FloatTensor)

    assert scale.shape[0] == 32
    assert scale.shape[1] == 76
    assert scale.shape[2] == 100
    assert isinstance(scale, torch.FloatTensor)

    layer.apply_weight_norm()
    layer.remove_weight_norm()


def test_ublock():
    inp1 = torch.rand(32, 50, 100)
    inp2 = torch.rand(32, 50, 50)
    nl = torch.rand(32)

    layer_film = FiLM(50, 100)
    layer = UBlock(50, 100, 2, [1, 2, 4, 8])

    scale, shift = layer_film(inp1, nl)
    o = layer(inp2, shift, scale)

    assert o.shape[0] == 32
    assert o.shape[1] == 100
    assert o.shape[2] == 100
    assert isinstance(o, torch.FloatTensor)

    layer.apply_weight_norm()
    layer.remove_weight_norm()


def test_dblock():
    inp = torch.rand(32, 50, 130)
    layer = DBlock(50, 100, 2)
    o = layer(inp)

    assert o.shape[0] == 32
    assert o.shape[1] == 100
    assert o.shape[2] == 65
    assert isinstance(o, torch.FloatTensor)

    layer.apply_weight_norm()
    layer.remove_weight_norm()


def test_wavegrad_forward():
    x = torch.rand(32, 1, 20 * 300)
    c = torch.rand(32, 80, 20)
    noise_scale = torch.rand(32)

    args = WavegradArgs(
        in_channels=80,
        out_channels=1,
        upsample_factors=[5, 5, 3, 2, 2],
        upsample_dilations=[[1, 2, 1, 2], [1, 2, 1, 2], [1, 2, 4, 8], [1, 2, 4, 8], [1, 2, 4, 8]],
    )
    config = WavegradConfig(model_params=args)
    model = Wavegrad(config)
    o = model.forward(x, c, noise_scale)

    assert o.shape[0] == 32
    assert o.shape[1] == 1
    assert o.shape[2] == 20 * 300
    assert isinstance(o, torch.FloatTensor)

    model.apply_weight_norm()
    model.remove_weight_norm()
