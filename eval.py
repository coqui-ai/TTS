import os
import torch
import time
import IPython

from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis

def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True, reference_wav=None, style_wav=None):
    t_1 = time.time()
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(
        model, text, CONFIG, use_cuda, ap, speaker_id, style_wav=None, truncated=False, enable_eos_bos_chars=CONFIG.enable_eos_bos_chars, use_griffin_lim=use_gl, reference_wav=reference_wav)
    # mel_postnet_spec = ap.denormalize(mel_postnet_spec.T)
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_postnet_spec.T).unsqueeze(0))
        waveform = waveform.flatten()
    if use_cuda:
        waveform = waveform
    # waveform = waveform.numpy()
    rtf = (time.time() - t_1) / (len(waveform) / ap.sample_rate)
    tps = (time.time() - t_1) / len(waveform)
    print(waveform.shape)
    print(" > Run-time: {}".format(time.time() - t_1))
    print(" > Real-time factor: {}".format(rtf))
    print(" > Time per step: {}".format(tps))
    IPython.display.display(IPython.display.Audio(waveform, rate=CONFIG.audio['sample_rate']))
    return alignment, mel_postnet_spec, stop_tokens, waveform


''' Runtime settings '''
use_cuda = True

# RUN_NAME = 'gts-blizzard-April-09-2021_05+51PM-a669a49'
RUN_NAME = 'capacitron-April-15-2021_06+38PM-26e9ee0'

# model paths
TTS_MODEL = os.path.join(r'/home/big-boy/Models/Blizzard', RUN_NAME, 'best_model.pth.tar')
TTS_CONFIG = os.path.join(r'/home/big-boy/Models/Blizzard', RUN_NAME, 'config.json')
VOCODER_MODEL = "data/vocoder_model.pth.tar"
VOCODER_CONFIG = "data/config_vocoder.json"

# load configs
TTS_CONFIG = load_config(TTS_CONFIG)
# VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# load the audio processor
# TTS_CONFIG.audio['stats_path'] = os.path.join(r'/home/big-boy/Models/Blizzard', 'blizzard-gts-March-17-2021_03+34PM-b4248b0', 'scale_stats.npy')

ap = AudioProcessor(**TTS_CONFIG.audio)

''' LOAD TTS MODEL '''

# multi speaker
speaker_id = None
speakers = []

# load the model
num_chars = len(phonemes) if TTS_CONFIG.use_phonemes else len(symbols)
model = setup_model(num_chars, len(speakers), TTS_CONFIG)

# load model state
cp = torch.load(TTS_MODEL, map_location=torch.device('cpu'))

# load the model
model.load_state_dict(cp['model'])
if use_cuda:
    model.cuda()
model.eval()

# set model stepsize
if 'r' in cp:
    model.decoder.set_r(cp['r'])

''' VOCODER '''

# from TTS.vocoder.utils.generic_utils import setup_generator

# # LOAD VOCODER MODEL
# vocoder_model = setup_generator(VOCODER_CONFIG)
# vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
# vocoder_model.remove_weight_norm()
# vocoder_model.inference_padding = 0

# ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])
# if use_cuda:
#     vocoder_model.cuda()
# vocoder_model.eval()

''' Run Inference '''

sentence = "Where there was a whole under the skirting board."
reference_path = '/home/big-boy/Data/blizzard2013/segmented/refs/unseen/usref-05.wav'
style_path = '/home/big-boy/Data/blizzard2013/segmented/refs/unseen/usref-05.wav'
align, spec, stop_tokens, wav = tts(model, sentence, TTS_CONFIG, use_cuda, ap, use_gl=True, figures=True,
                                    reference_wav=None, style_wav=None)

ap.save_wav(wav, 'usref04prior.wav')
