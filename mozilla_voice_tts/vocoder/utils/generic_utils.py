import re
import importlib
import numpy as np
from matplotlib import pyplot as plt

from mozilla_voice_tts.tts.utils.visual import plot_spectrogram


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


def setup_generator(c):
    print(" > Generator Model: {}".format(c.generator_model))
    MyModel = importlib.import_module('mozilla_voice_tts.vocoder.models.' +
                                      c.generator_model.lower())
    MyModel = getattr(MyModel, to_camel(c.generator_model))
    if c.generator_model in 'melgan_generator':
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
    if c.generator_model in 'multiband_melgan_generator':
        model = MyModel(
            in_channels=c.audio['num_mels'],
            out_channels=4,
            proj_kernel=7,
            base_channels=384,
            upsample_factors=c.generator_model_params['upsample_factors'],
            res_kernel=3,
            num_res_blocks=c.generator_model_params['num_res_blocks'])
    if c.generator_model in 'parallel_wavegan_generator':
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
    return model


def setup_discriminator(c):
    print(" > Discriminator Model: {}".format(c.discriminator_model))
    if 'parallel_wavegan' in c.discriminator_model:
        MyModel = importlib.import_module(
            'mozilla_voice_tts.vocoder.models.parallel_wavegan_discriminator')
    else:
        MyModel = importlib.import_module('mozilla_voice_tts.vocoder.models.' +
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
