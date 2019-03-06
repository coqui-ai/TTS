import io
import time
import librosa
import torch
import numpy as np
from .text import text_to_sequence, phoneme_to_sequence, sequence_to_phoneme
from .visual import visualize
from matplotlib import pylab as plt


def synthesis(m, s, CONFIG, use_cuda, ap):
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
    decoder_output, postnet_output, alignments, stop_tokens = m.inference(
        chars_var.long())
    postnet_output = postnet_output[0].data.cpu().numpy()
    decoder_output = decoder_output[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    if CONFIG.model == "Tacotron":
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_mel_spectrogram(postnet_output.T)
    wav = wav[:ap.find_endpoint(wav)]
    return wav, alignment, decoder_output, postnet_output, stop_tokens