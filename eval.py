from TTS.vocoder.utils.generic_utils import setup_generator
import os
import torch
import time
import IPython
import pandas as pd
from pathlib import Path
from os.path import join
import datetime

from TTS.tts.utils.generic_utils import setup_model
from TTS.utils.io import load_config
from TTS.tts.utils.text.symbols import symbols, phonemes
from TTS.utils.audio import AudioProcessor
from TTS.tts.utils.synthesis import synthesis

def tts(model, text, CONFIG, use_cuda, ap, use_gl, figures=True, reference_info=None, style_wav=None):
    t_1 = time.time()
    reference_wav = reference_info[0] if reference_info is not None else None
    reference_text = reference_info[1] if reference_info is not None else None
    waveform, alignment, mel_spec, mel_postnet_spec, stop_tokens, inputs = synthesis(
        model,
        text,
        CONFIG,
        use_cuda,
        ap,
        speaker_id,
        style_wav=style_wav,
        truncated=False,
        enable_eos_bos_chars=CONFIG.enable_eos_bos_chars,
        use_griffin_lim=use_gl,
        reference_wav=reference_wav,
        reference_text=reference_text
    )
    mel_postnet_spec = ap.denormalize(mel_postnet_spec.T)
    if not use_gl:
        waveform = vocoder_model.inference(torch.FloatTensor(mel_spec.T).unsqueeze(0))
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
    return alignment, mel_postnet_spec, stop_tokens, waveform


''' Runtime settings '''
use_cuda = True

''' Directory Mgmt '''

now = datetime.datetime.now()

RUN_NAME = 'capacitron-noPreemphasis-256-LRCompressed-May-17-2021_03+27PM-fca955c'
TEST_PATH = Path(join(r'/home/big-boy/Models/Blizzard/', RUN_NAME, 'TESTING'))
CURRENT_TEST_PATH = Path(join(TEST_PATH, now.strftime("%Y-%m-%d %H:%M:%S")))
TEST_PATH.mkdir(parents=True, exist_ok=True)

CURRENT_TEST_PATH.mkdir(parents=True, exist_ok=True)

# model paths
TTS_MODEL = join(r'/home/big-boy/Models/Blizzard', RUN_NAME, 'best_model.pth.tar')
TTS_CONFIG = join(r'/home/big-boy/Models/Blizzard', RUN_NAME, 'config.json')
VOCODER_MODEL = "/home/big-boy/Models/BlizzardVocoder/fullband-melgan-May-01-2021_11+30PM-2840cb5/checkpoint_150000.pth.tar"
VOCODER_CONFIG = "/home/big-boy/Models/BlizzardVocoder/fullband-melgan-May-01-2021_11+30PM-2840cb5/config.json"

# load configs
TTS_CONFIG = load_config(TTS_CONFIG)
VOCODER_CONFIG = load_config(VOCODER_CONFIG)

# load the audio processor
# TTS_CONFIG.audio['stats_path'] = join(r'/home/big-boy/Models/Blizzard', 'blizzard-gts-March-17-2021_03+34PM-b4248b0', 'scale_stats.npy')

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

# LOAD VOCODER MODEL
vocoder_model = setup_generator(VOCODER_CONFIG)
vocoder_model.load_state_dict(torch.load(VOCODER_MODEL, map_location="cpu")["model"])
vocoder_model.remove_weight_norm()
vocoder_model.inference_padding = 0

ap_vocoder = AudioProcessor(**VOCODER_CONFIG['audio'])
if use_cuda:
    vocoder_model.cuda()
vocoder_model.eval()

sentences = [
    "Sixty-Four comes asking for bread.",
    "Two seats were vacant.",
    "Let me help you with your baggage.",
    "The beauty of the sunset was obscured by the industrial cranes.",
    "He embraced his new life as an eggplant.",
    "Cursive writing is the best way to build a race track.",
    "They got there early, and they got really good seats.",
    "Your girlfriend bought your favorite cookie crisp cereal but forgot to get milk.",
    "A suit of armor provides excellent sun protection on hot days.",
    "She couldn't decide of the glass was half empty or half full so she drank it.",
    "Never underestimate the willingness of the greedy to throw you under the bus.",
    "She had a habit of taking showers in lemonade."
]

single_sentence = "Reality is the sum or aggregate of all that is real or existent within a system, as opposed to that which is only imaginary."

SAMPLE_FROM = 'prior' # 'prior' or 'posterior'
TEXT = 'single_sentence' # 'same_text' or 'sentences' or 'single_sentence'
TXT_DEPENDENCY = True

''' Run Inference '''
reference_df = pd.read_csv(Path('/home/big-boy/Data/blizzard2013/segmented/refs_metadata.csv'), header=None, names=['ID', 'Text'], sep='|', delimiter=None)
# reference_df = pd.read_csv(Path('/home/big-boy/Data/LJSpeech-1.1/refs_metadata.csv'), header=None, names=['ID', 'Text'], sep='|', delimiter=None)

for row in reference_df.iterrows():
    i = row[0]
    _id = row[1]['ID']
    reference_txt = row[1]['Text']

    sentence = sentences[i] if (TEXT == 'sentences') else reference_txt

    if TEXT == 'single_sentence':
        sentence = single_sentence

    reference_path = '/home/big-boy/Data/blizzard2013/segmented/refs/seen/{}.wav'.format(_id)
    reference_txt = reference_txt if TXT_DEPENDENCY else None

    refs = [reference_path, reference_txt] if SAMPLE_FROM == 'posterior' else None

    align, spec, stop_tokens, wav = tts(
        model,
        sentence,
        TTS_CONFIG,
        use_cuda,
        ap,
        use_gl=True,
        figures=True,
        reference_info=refs,
        style_wav=reference_path
    )

    file_handle = 'Prior' if (SAMPLE_FROM == 'prior') else 'Posterior'
    file_id = _id if TEXT == 'single_sentence' or TEXT == 'same_text' and SAMPLE_FROM != 'prior' else i

    ap.save_wav(wav, join(CURRENT_TEST_PATH, 'GMM_{}_{}.wav'.format(file_handle, file_id)))
