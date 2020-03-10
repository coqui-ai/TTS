import torch
import librosa
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from TTS.utils.text import phoneme_to_sequence, sequence_to_phoneme


def plot_alignment(alignment, info=None, fig_size=(16, 10), title=None):
    if isinstance(alignment, torch.Tensor):
        alignment_ = alignment.detach().cpu().numpy().squeeze()
    else:
        alignment_ = alignment
    fig, ax = plt.subplots(figsize=fig_size)
    im = ax.imshow(
        alignment_.T, aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    plt.xlabel(xlabel)
    plt.ylabel('Encoder timestep')
    # plt.yticks(range(len(text)), list(text))
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    return fig


def plot_spectrogram(linear_output, audio, fig_size=(16, 10)):
    if isinstance(linear_output, torch.Tensor):
        linear_output_ = linear_output.detach().cpu().numpy().squeeze()
    else:
        linear_output_ = linear_output
    spectrogram = audio._denormalize(linear_output_)  # pylint: disable=protected-access
    fig = plt.figure(figsize=fig_size)
    plt.imshow(spectrogram.T, aspect="auto", origin="lower")
    plt.colorbar()
    plt.tight_layout()
    return fig


def visualize(alignment, spectrogram_postnet, stop_tokens, text, hop_length, CONFIG, spectrogram=None, output_path=None):
    if spectrogram is not None:
        num_plot = 4
    else:
        num_plot = 3

    label_fontsize = 16
    fig = plt.figure(figsize=(8, 24))

    plt.subplot(num_plot, 1, 1)
    plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
    plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
    if CONFIG.use_phonemes:
        seq = phoneme_to_sequence(text, [CONFIG.text_cleaner], CONFIG.phoneme_language, CONFIG.enable_eos_bos_chars, tp=CONFIG.characters if 'characters' in CONFIG.keys() else None)
        text = sequence_to_phoneme(seq, tp=CONFIG.characters if 'characters' in CONFIG.keys() else None)
        print(text)

    plt.yticks(range(len(text)), list(text))
    plt.colorbar()

    stop_tokens = stop_tokens.squeeze().detach().to('cpu').numpy()
    plt.subplot(num_plot, 1, 2)
    plt.plot(range(len(stop_tokens)), list(stop_tokens))

    plt.subplot(num_plot, 1, 3)
    librosa.display.specshow(spectrogram_postnet.T, sr=CONFIG.audio['sample_rate'],
                             hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.xlabel("Time", fontsize=label_fontsize)
    plt.ylabel("Hz", fontsize=label_fontsize)
    plt.tight_layout()
    plt.colorbar()

    if spectrogram is not None:
        plt.subplot(num_plot, 1, 4)
        librosa.display.specshow(spectrogram.T, sr=CONFIG.audio['sample_rate'],
                                 hop_length=hop_length, x_axis="time", y_axis="linear")
        plt.xlabel("Time", fontsize=label_fontsize)
        plt.ylabel("Hz", fontsize=label_fontsize)
        plt.tight_layout()
        plt.colorbar()

    if output_path:
        print(output_path)
        fig.savefig(output_path)
        plt.close()
