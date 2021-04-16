from TTS.utils.io import load_config
from TTS.tts.models.tacotron import Tacotron
import torch
from IPython.display import Audio
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.tts.utils.visual import visualize
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text import text_to_sequence, cleaners
from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.data import *
from TTS.tts.layers import *
import librosa.display
import librosa
import IPython
from matplotlib import pylab as plt
from collections import OrderedDict
import numpy as np
import time
import re
import io
import sys
import os

#pip3 install --user numpy


from collections import OrderedDict
#from matplotlib import pylab as plt

import torch
#To install with CUDA 9.2. This worked for me
#https://developer.nvidia.com/cuda-92-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exenetwork
#pip3 install --user --no-cache-dir torch==1.2.0+cu92 torchvision==0.4.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html

#For trying CUDA 10.0. This didn't work for me
#https://developer.nvidia.com/cuda-10.0-download-archive?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exelocal
#pip3 install --no-cache-dir --user torch==1.2.0 torchvision==0.4.0 -f https://download.pytorch.org/whl/cu100/torch_stable.html


TTS_PATH = os.path.join(r'/home/big-boy/projects/Capacitron')
WAVERNN_PATH = os.path.join(TTS_PATH, 'WaveRNN')

#%pylab inline
#rcParams["figure.figsize"] = (16,5)

# add libraries into environment
#import importlib
#importlib.import_module('TTS')

sys.path.append(TTS_PATH) # set this if TTS is not installed globally
sys.path.append(WAVERNN_PATH) # set this if TTS is not installed globally


#from utils.visual import visualize


#import IPython
#from IPython.display import Audio
#pip3 install --user ipython

#os.environ['CUDA_VISIBLE_DEVICES']='1'
#os.environ['OMP_NUM_THREADS']='1'

iscuda = torch.cuda.is_available()
print('torch.cuda.is_available()=' + str(iscuda))

runcounter = 0
def tts(model, text, CONFIG, use_cuda, ap, use_gl, speaker_id=None, figures=True):
    global runcounter
    t_1 = time.time()
    submatch = re.sub(r'\s+', ' ', text)
    file_namematch = re.search(r'([^\s]+\s?\d+)', submatch)
    if file_namematch:
        file_name = file_namematch.group(0) + '_' + str(runcounter) + '.wav'
    else:
        file_name = 'tempout_' + str(runcounter) + '.wav'
    runcounter += 1
    waveform, alignment, decoder_output, postnet_output, stop_tokens, inputs = synthesis(model, text, CONFIG, use_cuda, ap, use_griffin_lim=use_gl, truncated=False)
    if CONFIG.model == "Tacotron" and not use_gl:
        postnet_output = ap.out_linear_to_mel(postnet_output.T).T
    if not use_gl:
        waveform = wavernn.generate(torch.FloatTensor(postnet_output.T).unsqueeze(0).cuda(), batched=batched_wavernn, target=11000, overlap=550)

    print(" >  Run-time: {}".format(time.time() - t_1))
    #if figures:
    #    visualize(alignment, mel_postnet_spec, stop_tokens, text, ap.hop_length, CONFIG, mel_spec)
    #IPython.display.display(Audio(waveform, rate=CONFIG.audio['sample_rate']))
    os.makedirs(OUT_FOLDER, exist_ok=True)

    out_path = os.path.join(OUT_FOLDER, file_name)
    ap.save_wav(waveform, out_path)
    return alignment, postnet_output, stop_tokens, waveform


# Set constants
ROOT_PATH = TTS_PATH
MODEL_PATH = os.path.join(r'/home/big-boy/Models/Blizzard/', 'capacitron-April-06-2021_05+48PM-26e9ee0', 'best_model.pth.tar')
CONFIG_PATH = os.path.join(r'/home/big-boy/projects/Capacitron/TTS/tts/configs', 'capacitron_blizzard.json')
OUT_FOLDER = os.path.join(ROOT_PATH, 'AudioSamples/benchmark_samples/')
CONFIG = load_config(CONFIG_PATH)
# VOCODER_MODEL_PATH = os.path.join(r'C:\Users\sokka\Documents\tts\wavernn_mold\wavernn_mold_8a1c152', 'checkpoint_433000.pth.tar')
# VOCODER_CONFIG_PATH = os.path.join(r'C:\Users\sokka\Documents\tts\wavernn_mold\wavernn_mold_8a1c152', 'config.json')
# VOCODER_CONFIG = load_config(VOCODER_CONFIG_PATH)
use_cuda = True

# tts_pretrained_model = '/home/big-boy/Models/Blizzard/blizzard-gts-March-17-2021_03+34PM-b4248b0/best_model.pth.tar'
# # tts_pretrained_model_config = '/home/big-boy/projects/Capacitron/TTS/tts/configs/gst_blizzard.json'

# Set some config fields manually for testing
# CONFIG.windowing = False
# CONFIG.prenet_dropout = False
# CONFIG.separate_stopnet = True
# CONFIG.stopnet = True

# Set the vocoder
use_gl = True # use GL if True
batched_wavernn = False    # use batched wavernn inference if True
num_speakers = 1

# LOAD TTS MODEL

# load the model
num_chars = len(phonemes) if CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, num_speakers, CONFIG)

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


# LOAD WAVERNN
if use_gl == False:
    # from WaveRNN.models.wavernn import Model
    bits = 10

    wavernn = Model(
        rnn_dims=512,
        fc_dims=512,
        mode="mold",
        pad=2,
        upsample_factors=VOCODER_CONFIG.upsample_factors,  # set this depending on dataset
        feat_dims=VOCODER_CONFIG.audio["num_mels"],
        compute_dims=128,
        res_out_dims=128,
        res_blocks=10,
        hop_length=ap.hop_length,
        sample_rate=ap.sample_rate,
    ).cuda()

    check = torch.load(VOCODER_MODEL_PATH)
    wavernn.load_state_dict(check['model'])
    if use_cuda:
        wavernn.cuda()
    wavernn.eval()
    print(check['step'])


illegalchars_exclusive = re.compile(r'[^\w\d\.\,\;\!\?\s]')
repitiion = re.compile(r'\s{2,}')
def custom_text_fix(sentence):
    global illegalchars_exclusive
    global repitiion
    newsentance = illegalchars_exclusive.sub(' ', sentence)
    newsentance = repitiion.sub(' ', newsentance)
    return newsentance


model.eval()
model.decoder.max_decoder_steps = 2000
speaker_id = 0

sentences = ["Bill got in the habit of asking himself “Is that thought true?” And if he wasn’t absolutely certain it was, he just let it go."]

for sentence in sentences:
    sentence = custom_text_fix(sentence)
    sentence = cleaners.english_cleaners(sentence)
    alignment, postnet_output, stop_tokens, waveform = tts(model, sentence, CONFIG, use_cuda, ap, speaker_id=speaker_id, use_gl=use_gl, figures=True)

def main():
    pass


if __name__ == 'main':
    main()
