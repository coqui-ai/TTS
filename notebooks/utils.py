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

#     mel = np.zeros([seq.shape[0], CONFIG.num_mels, 1], dtype=np.float32)

    if use_cuda:
        chars_var = torch.autograd.Variable(
            torch.from_numpy(seq), volatile=True).unsqueeze(0).cuda()
#         mel_var = torch.autograd.Variable(torch.from_numpy(mel).type(torch.cuda.FloatTensor), volatile=True).cuda()
    else:
        chars_var = torch.autograd.Variable(
            torch.from_numpy(seq), volatile=True).unsqueeze(0)
#         mel_var = torch.autograd.Variable(torch.from_numpy(mel).type(torch.FloatTensor), volatile=True)

    mel_out, linear_out, alignments = m.forward(chars_var)
    linear_out = linear_out[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    spec = ap._denormalize(linear_out)
    wav = ap.inv_spectrogram(linear_out.T)
    wav = wav[:ap.find_endpoint(wav)]
    out = io.BytesIO()
    ap.save_wav(wav, out)
    return wav, alignment, spec


def visualize(alignment, spectrogram, CONFIG):
    label_fontsize = 16
    plt.figure(figsize=(16, 16))

    plt.subplot(2, 1, 1)
    plt.imshow(alignment.T, aspect="auto", origin="lower", interpolation=None)
    plt.xlabel("Decoder timestamp", fontsize=label_fontsize)
    plt.ylabel("Encoder timestamp", fontsize=label_fontsize)
    plt.colorbar()

    plt.subplot(2, 1, 2)
    librosa.display.specshow(spectrogram.T, sr=CONFIG.sample_rate,
                             hop_length=hop_length, x_axis="time", y_axis="linear")
    plt.xlabel("Time", fontsize=label_fontsize)
    plt.ylabel("Hz", fontsize=label_fontsize)
    plt.tight_layout()
    plt.colorbar()
