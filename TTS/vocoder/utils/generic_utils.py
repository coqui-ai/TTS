import re
import torch
import importlib
import numpy as np
from matplotlib import pyplot as plt

from TTS.tts.utils.visual import plot_spectrogram


def interpolate_vocoder_input(scale_factor, spec):
    """Interpolate spectrogram by the scale factor.
    It is mainly used to match the sampling rates of
    the tts and vocoder models.

    Args:
        scale_factor (float): scale factor to interpolate the spectrogram
        spec (np.array): spectrogram to be interpolated

    Returns:
        torch.tensor: interpolated spectrogram.
    """
    print(" > before interpolation :", spec.shape)
    spec = torch.tensor(spec).unsqueeze(0).unsqueeze(0)  # pylint: disable=not-callable
    spec = torch.nn.functional.interpolate(spec,
                                           scale_factor=scale_factor,
                                           recompute_scale_factor=True,
                                           mode='bilinear',
                                           align_corners=False).squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


def plot_results(y_hat, y, ap, global_step, name_prefix):
    """ Plot vocoder model results """

    # select an instance from batch
    y_hat = y_hat[0].squeeze(0).detach().cpu().numpy()
    y = y[0].squeeze(0).detach().cpu().numpy()

    spec_fake = ap.melspectrogram(y_hat).T
    spec_real = ap.melspectrogram(y).T
    spec_diff = np.abs(spec_fake - spec_real)

    # plot figure and save it
    fig_wave = plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(y)
    plt.title("groundtruth speech")
    plt.subplot(2, 1, 2)
    plt.plot(y_hat)
    plt.title(f"generated speech @ {global_step} steps")
    plt.tight_layout()
    plt.close()

    figures = {
        name_prefix + "spectrogram/fake": plot_spectrogram(spec_fake),
        name_prefix + "spectrogram/real": plot_spectrogram(spec_real),
        name_prefix + "spectrogram/diff": plot_spectrogram(spec_diff),
        name_prefix + "speech_comparison": fig_wave,
    }
    return figures


def to_camel(text):
    text = text.capitalize()
    return re.sub(r'(?!^)_([a-zA-Z])', lambda m: m.group(1).upper(), text)


def setup_wavernn(c):
    print(" > Model: WaveRNN")
    MyModel = importlib.import_module("TTS.vocoder.models.wavernn")
    MyModel = getattr(MyModel, "WaveRNN")
    model = MyModel(
        rnn_dims=c.wavernn_model_params['rnn_dims'],
        fc_dims=c.wavernn_model_params['fc_dims'],
        mode=c.mode,
        mulaw=c.mulaw,
        pad=c.padding,
        use_aux_net=c.wavernn_model_params['use_aux_net'],
        use_upsample_net=c.wavernn_model_params['use_upsample_net'],
        upsample_factors=c.wavernn_model_params['upsample_factors'],
        feat_dims=c.audio['num_mels'],
        compute_dims=c.wavernn_model_params['compute_dims'],
        res_out_dims=c.wavernn_model_params['res_out_dims'],
        num_res_blocks=c.wavernn_model_params['num_res_blocks'],
        hop_length=c.audio["hop_length"],
        sample_rate=c.audio["sample_rate"],
    )
    return model


def setup_generator(c):
    print(" > Generator Model: {}".format(c.generator_model))
    MyModel = importlib.import_module('TTS.vocoder.models.' +
                                      c.generator_model.lower())
    MyModel = getattr(MyModel, to_camel(c.generator_model))
    if c.generator_model.lower() in 'melgan_generator':
        model = MyModel(
            in_channels=c.audio['num_mels'],
            out_channels=1,
            proj_kernel=7,
            base_channels=512,
            upsample_factors=c.generator_model_params['upsample_factors'],
            res_kernel=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'])
    if c.generator_model in 'melgan_fb_generator':
        pass
    if c.generator_model.lower() in 'multiband_melgan_generator':
        model = MyModel(
            in_channels=c.audio['num_mels'],
            out_channels=4,
            proj_kernel=7,
            base_channels=384,
            upsample_factors=c.generator_model_params['upsample_factors'],
            res_kernel=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'])
    if c.generator_model.lower() in 'fullband_melgan_generator':
        model = MyModel(
            in_channels=c.audio['num_mels'],
            out_channels=1,
            proj_kernel=7,
            base_channels=512,
            upsample_factors=c.generator_model_params['upsample_factors'],
            res_kernel=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'])
    if c.generator_model.lower() in 'parallel_wavegan_generator':
        model = MyModel(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'],
            stacks=c.generator_model_params['stacks'],
            res_channels=64,
            gate_channels=128,
            skip_channels=64,
            aux_channels=c.audio['num_mels'],
            dropout=0.0,
            bias=True,
            use_weight_norm=True,
            upsample_factors=c.generator_model_params['upsample_factors'])
    if c.generator_model.lower() in 'wavegrad':
        model = MyModel(
            in_channels=c['audio']['num_mels'],
            out_channels=1,
            use_weight_norm=c['model_params']['use_weight_norm'],
            x_conv_channels=c['model_params']['x_conv_channels'],
            y_conv_channels=c['model_params']['y_conv_channels'],
            dblock_out_channels=c['model_params']['dblock_out_channels'],
            ublock_out_channels=c['model_params']['ublock_out_channels'],
            upsample_factors=c['model_params']['upsample_factors'],
            upsample_dilations=c['model_params']['upsample_dilations'])
    return model


def setup_discriminator(c):
    print(" > Discriminator Model: {}".format(c.discriminator_model))
    if 'parallel_wavegan' in c.discriminator_model:
        MyModel = importlib.import_module(
            'TTS.vocoder.models.parallel_wavegan_discriminator')
    else:
        MyModel = importlib.import_module('TTS.vocoder.models.' +
                                          c.discriminator_model.lower())
    MyModel = getattr(MyModel, to_camel(c.discriminator_model.lower()))
    if c.discriminator_model in 'random_window_discriminator':
        model = MyModel(
            cond_channels=c.audio['num_mels'],
            hop_length=c.audio['hop_length'],
            uncond_disc_donwsample_factors=c.
            discriminator_model_params['uncond_disc_donwsample_factors'],
            cond_disc_downsample_factors=c.
            discriminator_model_params['cond_disc_downsample_factors'],
            cond_disc_out_channels=c.
            discriminator_model_params['cond_disc_out_channels'],
            window_sizes=c.discriminator_model_params['window_sizes'])
    if c.discriminator_model in 'melgan_multiscale_discriminator':
        model = MyModel(
            in_channels=1,
            out_channels=1,
            kernel_sizes=(5, 3),
            base_channels=c.discriminator_model_params['base_channels'],
            max_channels=c.discriminator_model_params['max_channels'],
            downsample_factors=c.
            discriminator_model_params['downsample_factors'])
    if c.discriminator_model == 'residual_parallel_wavegan_discriminator':
        model = MyModel(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            num_layers=c.discriminator_model_params['num_layers'],
            stacks=c.discriminator_model_params['stacks'],
            res_channels=64,
            gate_channels=128,
            skip_channels=64,
            dropout=0.0,
            bias=True,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.2},
        )
    if c.discriminator_model == 'parallel_wavegan_discriminator':
        model = MyModel(
            in_channels=1,
            out_channels=1,
            kernel_size=3,
            num_layers=c.discriminator_model_params['num_layers'],
            conv_channels=64,
            dilation_factor=1,
            nonlinear_activation="LeakyReLU",
            nonlinear_activation_params={"negative_slope": 0.2},
            bias=True
        )
    return model


# def check_config(c):
#     c = None
#     pass
