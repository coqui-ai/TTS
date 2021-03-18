# %load_ext autoreload
# %autoreload 2
from TTS.tts.utils.text.symbols import make_symbols, symbols, phonemes
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.visual import plot_alignment
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text import text_to_sequence
from TTS.utils.io import load_config
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.tts.layers import *
import librosa.display
import librosa
import os
import sys
import torch
import time
import numpy as np
from matplotlib import pylab as plt

# %pylab inline
plt.rcParams["figure.figsize"] = (16, 5)

print(torch.cuda.is_available())


# import IPython
# from IPython.display import Audio

os.environ['CUDA_VISIBLE_DEVICES'] = '1'


def tts(model, text, CONFIG, use_cuda, ap):
    t_1 = time.time()
    # run the model
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(
        model, text, CONFIG, use_cuda, ap, speaker_id, None, False, CONFIG.enable_eos_bos_chars, True)
    if CONFIG.model == "Tacotron" and not use_gl:
        mel_postnet_spec = ap.out_linear_to_mel(mel_postnet_spec.T).T
    # plotting
    attn_score = alignment_diagonal_score(
        torch.FloatTensor(alignment).unsqueeze(0))
    print(f" > {text}")
    IPython.display.display(IPython.display.Audio(
        waveform, rate=ap.sample_rate))
    fig = plot_alignment(alignment, fig_size=(8, 5))
    IPython.display.display(fig)
    # saving results
    os.makedirs(OUT_FOLDER, exist_ok=True)
    file_name = text[:200].replace(" ", "_").replace(".", "") + ".wav"
    out_path = os.path.join(OUT_FOLDER, file_name)
    ap.save_wav(waveform, out_path)
    return attn_score


# Set constants
ROOT_PATH = '/home/big-boy/Models/Blizzard/blizzard-gts-March-15-2021_05+24PM-b4248b0'
MODEL_PATH = ROOT_PATH + '/best_model.pth.tar'
CONFIG_PATH = 'TTS/tts/configs/gst_blizzard.json'
OUT_FOLDER = './hard_sentences/'
CONFIG = load_config(CONFIG_PATH)
SENTENCES_PATH = 'sentences.txt'
use_cuda = True

# Set some config fields manually for testing
# CONFIG.windowing = False
# CONFIG.prenet_dropout = False
# CONFIG.separate_stopnet = True
CONFIG.use_forward_attn = False
# CONFIG.forward_attn_mask = True
# CONFIG.stopnet = True

# LOAD TTS MODEL

# multi speaker
if CONFIG.use_speaker_embedding:
    speakers = json.load(open(f"{ROOT_PATH}/speakers.json", 'r'))
    speakers_idx_to_id = {v: k for k, v in speakers.items()}
else:
    speakers = []
    speaker_id = None

# if the vocabulary was passed, replace the default
if 'characters' in CONFIG.keys():
    symbols, phonemes = make_symbols(**CONFIG.characters)

# load the model
num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), CONFIG)

# load the audio processor
ap = AudioProcessor(**CONFIG.audio)


# load model state
if use_cuda:
    cp = torch.load(MODEL_PATH)
else:
    cp = torch.load(MODEL_PATH, map_location=lambda storage, loc: storage)

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()
print(cp['step'])
print(cp['r'])

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

model.decoder.max_decoder_steps = 3000
attn_scores = []
with open(SENTENCES_PATH, 'r') as f:
    for text in f:
        attn_score = tts(model, text, CONFIG, use_cuda, ap)
        attn_scores.append(attn_score)

print(np.mean(attn_scores))
