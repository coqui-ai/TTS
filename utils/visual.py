import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def plot_alignment(alignment, info=None):
    fig, ax = plt.subplots(figsize=(16, 10))
    im = ax.imshow(
        alignment.T, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    return fig


def plot_spectrogram(linear_output, audio):
    spectrogram = audio._denormalize(linear_output)
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    return fig
