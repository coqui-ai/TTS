import io
import time
import librosa
import torch
import numpy as np
from .text import text_to_sequence, phoneme_to_sequence, sequence_to_phoneme
from .visual import visualize
from matplotlib import pylab as plt


def synthesis(model, text, CONFIG, use_cuda, ap, truncated=False, enable_eos_bos_chars=False, trim_silence=False):
    """Synthesize voice for the given text.

        Args:
            model (TTS.models): model to synthesize.
            text (str): target text
            CONFIG (dict): config dictionary to be loaded from config.json.
            use_cuda (bool): enable cuda.
            ap (TTS.utils.audio.AudioProcessor): audio processor to process
                model outputs.
            truncated (bool): keep model states after inference. It can be used
                for continuous inference at long texts.
            enable_eos_bos_chars (bool): enable special chars for end of sentence and start of sentence.
            trim_silence (bool): trim silence after synthesis.
    """
    # preprocess the given text
    text_cleaner = [CONFIG.text_cleaner]
    if CONFIG.use_phonemes:
        seq = np.asarray(
            phoneme_to_sequence(text, text_cleaner, CONFIG.phoneme_language, enable_eos_bos_chars),
            dtype=np.int32)
    else:
        seq = np.asarray(text_to_sequence(text, text_cleaner), dtype=np.int32)
    chars_var = torch.from_numpy(seq).unsqueeze(0)
    # synthesize voice
    if use_cuda:
        chars_var = chars_var.cuda()
    if truncated:
        decoder_output, postnet_output, alignments, stop_tokens = model.inference_truncated(
            chars_var.long())
    else:
        decoder_output, postnet_output, alignments, stop_tokens = model.inference(
            chars_var.long())
    # convert outputs to numpy
    postnet_output = postnet_output[0].data.cpu().numpy()
    decoder_output = decoder_output[0].data.cpu().numpy()
    alignment = alignments[0].cpu().data.numpy()
    # plot results
    if CONFIG.model == "Tacotron":
        wav = ap.inv_spectrogram(postnet_output.T)
    else:
        wav = ap.inv_mel_spectrogram(postnet_output.T)
    # trim silence
    if trim_silence:
        wav = wav[:ap.find_endpoint(wav)]
    return wav, alignment, decoder_output, postnet_output, stop_tokens