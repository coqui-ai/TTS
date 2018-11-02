import numpy as np
import librosa
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
    # plt.yticks(range(len(text)), list(text))
    plt.tight_layout()
    return fig


def plot_spectrogram(linear_output, audio):
    spectrogram = audio._denormalize(linear_output)
    fig = plt.figure(figsize=(16, 10))
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    return fig


def visualize(alignment, spectrogram, stop_tokens, text, hop_length, CONFIG):
    label_fontsize = 16
    plt.figure(figsize=(16, 32))

    plt.subplot(3, 1, 1)
    plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
    plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
    plt.yticks(range(len(text)), list(text))
    plt.colorbar()
    
    stop_tokens = stop_tokens.squeeze().detach().to('cpu').numpy()
    plt.subplot(3, 1, 2)
    plt.plot(range(len(stop_tokens)), list(stop_tokens))

    plt.subplot(3, 1, 3)
    librosa.display.specshow(spectrogram.T, sr=CONFIG.audio['sample_rate'],
                             hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.xlabel("Time", fontsize=label_fontsize)
    plt.ylabel("Hz", fontsize=label_fontsize)

    plt.tight_layout()
    plt.colorbar()
