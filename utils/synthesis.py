import io
import time
import librosa
import torch
import numpy as np
from .text import text_to_sequence, phoneme_to_sequence, sequence_to_phoneme
from .visual import visualize
from matplotlib import pylab as plt


def synthesis(m, s, CONFIG, use_cuda, ap):
    """ Given the text, synthesising the audio """
    text_cleaner = [CONFIG.text_cleaner]
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(s, text_cleaner, CONFIG.phoneme_language),
            dtype=np.int32)
    else:
        seq = np.asarray(text_to_sequence(s, text_cleaner), dtype=np.int32)
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    if use_cuda:
        chars_var = chars_var.cuda()
    mel_spec, linear_spec, alignments, stop_tokens = m.forward(
        chars_var.long())
    linear_spec = linear_spec[0].data.cpu().numpy()
    mel_spec = mel_spec[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    wav = ap.inv_spectrogram(linear_spec.T)
    wav = wav[:ap.find_endpoint(wav)]
    return wav, alignment, linear_spec, mel_spec, stop_tokens