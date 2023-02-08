import librosa
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LogNorm

matplotlib.use("Agg")


def plot_alignment(alignment, info=None, fig_size=(16, 10), title=None, output_fig=False, plot_log=False):
    if isinstance(alignment, torch.Tensor):
        alignment_ = alignment.detach().cpu().numpy().squeeze()
    else:
        alignment_ = alignment
    alignment_ = alignment_.astype(np.float32) if alignment_.dtype == np.float16 else alignment_
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        alignment_.T, aspect="auto", origin="lower", interpolation="none", norm=LogNorm() if plot_log else None
    )
    fig.colorbar(im, ax=ax)
    xlabel = "Decoder timestep"
    if info is not None:
        xlabel += "\n\n" + info
    plt.xlabel(xlabel)
    plt.ylabel("Encoder timestep")
    # plt.yticks(range(len(text)), list(text))
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    if not output_fig:
        plt.close()
    return fig


def plot_spectrogram(spectrogram, ap=None, fig_size=(16, 10), output_fig=False):
    if isinstance(spectrogram, torch.Tensor):
        spectrogram_ = spectrogram.detach().cpu().numpy().squeeze().T
    else:
        spectrogram_ = spectrogram.T
    spectrogram_ = spectrogram_.astype(np.float32) if spectrogram_.dtype == np.float16 else spectrogram_
    if ap is not None:
        spectrogram_ = ap.denormalize(spectrogram_)  # pylint: disable=protected-access
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram_, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    if not output_fig:
        plt.close()
    return fig


def plot_pitch(pitch, spectrogram, ap=None, fig_size=(30, 10), output_fig=False):
    """Plot pitch curves on top of the spectrogram.

    Args:
        pitch (np.array): Pitch values.
        spectrogram (np.array): Spectrogram values.

    Shapes:
        pitch: :math:`(T,)`
        spec: :math:`(C, T)`
    """

    if isinstance(spectrogram, torch.Tensor):
        spectrogram_ = spectrogram.detach().cpu().numpy().squeeze().T
    else:
        spectrogram_ = spectrogram.T
    spectrogram_ = spectrogram_.astype(np.float32) if spectrogram_.dtype == np.float16 else spectrogram_
    if ap is not None:
        spectrogram_ = ap.denormalize(spectrogram_)  # pylint: disable=protected-access

    old_fig_size = plt.rcParams["figure.figsize"]
    if fig_size is not None:
        plt.rcParams["figure.figsize"] = fig_size

    fig, ax = plt.subplots()

    ax.imshow(spectrogram_, aspect="auto", origin="lower")
    ax.set_xlabel("time")
    ax.set_ylabel("spec_freq")

    ax2 = ax.twinx()
    ax2.plot(pitch, linewidth=5.0, color="red")
    ax2.set_ylabel("F0")

    plt.rcParams["figure.figsize"] = old_fig_size
    if not output_fig:
        plt.close()
    return fig


def plot_avg_pitch(pitch, chars, fig_size=(30, 10), output_fig=False):
    """Plot pitch curves on top of the input characters.

    Args:
        pitch (np.array): Pitch values.
        chars (str): Characters to place to the x-axis.

    Shapes:
        pitch: :math:`(T,)`
    """
    old_fig_size = plt.rcParams["figure.figsize"]
    if fig_size is not None:
        plt.rcParams["figure.figsize"] = fig_size

    fig, ax = plt.subplots()

    x = np.array(range(len(chars)))
    my_xticks = chars
    plt.xticks(x, my_xticks)

    ax.set_xlabel("characters")
    ax.set_ylabel("freq")

    ax2 = ax.twinx()
    ax2.plot(pitch, linewidth=5.0, color="red")
    ax2.set_ylabel("F0")

    plt.rcParams["figure.figsize"] = old_fig_size
    if not output_fig:
        plt.close()
    return fig


def plot_avg_energy(energy, chars, fig_size=(30, 10), output_fig=False):
    """Plot energy curves on top of the input characters.

    Args:
        energy (np.array): energy values.
        chars (str): Characters to place to the x-axis.

    Shapes:
        energy: :math:`(T,)`
    """
    old_fig_size = plt.rcParams["figure.figsize"]
    if fig_size is not None:
        plt.rcParams["figure.figsize"] = fig_size

    fig, ax = plt.subplots()

    x = np.array(range(len(chars)))
    my_xticks = chars
    plt.xticks(x, my_xticks)

    ax.set_xlabel("characters")
    ax.set_ylabel("freq")

    ax2 = ax.twinx()
    ax2.plot(energy, linewidth=5.0, color="red")
    ax2.set_ylabel("energy")

    plt.rcParams["figure.figsize"] = old_fig_size
    if not output_fig:
        plt.close()
    return fig


def visualize(
    alignment,
    postnet_output,
    text,
    hop_length,
    CONFIG,
    tokenizer,
    stop_tokens=None,
    decoder_output=None,
    output_path=None,
    figsize=(8, 24),
    output_fig=False,
):
    """Intended to be used in Notebooks."""

    if decoder_output is not None:
        num_plot = 4
    else:
        num_plot = 3

    label_fontsize = 16
    fig = plt.figure(figsize=figsize)

    plt.subplot(num_plot, 1, 1)
    plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
    plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
    # compute phoneme representation and back
    if CONFIG.use_phonemes:
        seq = tokenizer.text_to_ids(text)
        text = tokenizer.ids_to_text(seq)
        print(text)
    plt.yticks(range(len(text)), list(text))
    plt.colorbar()

    if stop_tokens is not None:
        # plot stopnet predictions
        plt.subplot(num_plot, 1, 2)
        plt.plot(range(len(stop_tokens)), list(stop_tokens))

    # plot postnet spectrogram
    plt.subplot(num_plot, 1, 3)
    librosa.display.specshow(
        postnet_output.T,
        sr=CONFIG.audio["sample_rate"],
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear",
        fmin=CONFIG.audio["mel_fmin"],
        fmax=CONFIG.audio["mel_fmax"],
    )

    plt.xlabel("Time", fontsize=label_fontsize)
    plt.ylabel("Hz", fontsize=label_fontsize)
    plt.tight_layout()
    plt.colorbar()

    if decoder_output is not None:
        plt.subplot(num_plot, 1, 4)
        librosa.display.specshow(
            decoder_output.T,
            sr=CONFIG.audio["sample_rate"],
            hop_length=hop_length,
            x_axis="time",
            y_axis="linear",
            fmin=CONFIG.audio["mel_fmin"],
            fmax=CONFIG.audio["mel_fmax"],
        )
        plt.xlabel("Time", fontsize=label_fontsize)
        plt.ylabel("Hz", fontsize=label_fontsize)
        plt.tight_layout()
        plt.colorbar()

    if output_path:
        print(output_path)
        fig.savefig(output_path)
        plt.close()

    if not output_fig:
        plt.close()
