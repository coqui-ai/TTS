from typing import Dict

import numpy as np
import torch
from matplotlib import pyplot as plt

from TTS.tts.utils.visual import plot_spectrogram
from TTS.utils.audio import AudioProcessor


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
    spec = torch.nn.functional.interpolate(
        spec, scale_factor=scale_factor, recompute_scale_factor=True, mode="bilinear", align_corners=False
    ).squeeze(0)
    print(" > after interpolation :", spec.shape)
    return spec


def plot_results(y_hat: torch.tensor, y: torch.tensor, ap: AudioProcessor, name_prefix: str = None) -> Dict:
    """Plot the predicted and the real waveform and their spectrograms.

    Args:
        y_hat (torch.tensor): Predicted waveform.
        y (torch.tensor): Real waveform.
        ap (AudioProcessor): Audio processor used to process the waveform.
        name_prefix (str, optional): Name prefix used to name the figures. Defaults to None.

    Returns:
        Dict: output figures keyed by the name of the figures.
    """ """Plot vocoder model results"""
    if name_prefix is None:
        name_prefix = ""

    # select an instance from batch
    y_hat = y_hat[0].squeeze().detach().cpu().numpy()
    y = y[0].squeeze().detach().cpu().numpy()

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
    plt.title("generated speech")
    plt.tight_layout()
    plt.close()

    figures = {
        name_prefix + "spectrogram/fake": plot_spectrogram(spec_fake),
        name_prefix + "spectrogram/real": plot_spectrogram(spec_real),
        name_prefix + "spectrogram/diff": plot_spectrogram(spec_diff),
        name_prefix + "speech_comparison": fig_wave,
    }
    return figures
