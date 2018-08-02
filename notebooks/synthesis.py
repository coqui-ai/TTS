import io
import librosa
import torch
import numpy as np
from TTS.utils.text import text_to_sequence
from matplotlib import pylab as plt

hop_length = 250


def create_speech(m, s, CONFIG, use_cuda, ap):
    text_cleaner = [CONFIG.text_cleaner]
    seq = np.array(text_to_sequence(s, text_cleaner))
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    mel_out, linear_out, alignments, stop_tokens = m.forward(chars_var.long())
    linear_out = linear_out[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    spec = ap._denormalize(linear_out)
    wav = ap.inv_spectrogram(linear_out.T)
    wav = wav[:ap.find_endpoint(wav)]
    out = io.BytesIO()
    ap.save_wav(wav, out)
    return wav, alignment, spec, stop_tokens


def visualize(alignment, spectrogram, stop_tokens, CONFIG):
    label_fontsize = 16
    plt.figure(figsize=(16, 24))

    plt.subplot(3, 1, 1)
    plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
    plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
    plt.colorbar()

    stop_tokens = stop_tokens.squeeze().detach().to('cpu').numpy()
    plt.subplot(3, 1, 2)
    plt.plot(range(len(stop_tokens)), list(stop_tokens))

    plt.subplot(3, 1, 3)
    librosa.display.specshow(
        spectrogram.T,
        sr=CONFIG.sample_rate,
        hop_length=hop_length,
        x_axis="time",
        y_axis="linear")
    plt.xlabel("Time", fontsize=label_fontsize)
    plt.ylabel("Hz", fontsize=label_fontsize)
    plt.tight_layout()
    plt.colorbar()
