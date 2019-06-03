import io
import os
<<<<<<< HEAD

import numpy as np
import torch
=======
import sys

import numpy as np
import torch

from models.tacotron import Tacotron
from utils.audio import AudioProcessor
from utils.generic_utils import load_config, setup_model
from utils.text import phoneme_to_sequence, phonemes, symbols, text_to_sequence, sequence_to_phoneme

import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
>>>>>>> dev-tacotron2

from models.tacotron import Tacotron
from utils.audio import AudioProcessor
from utils.generic_utils import load_config
from utils.text import phoneme_to_sequence, phonemes, symbols, text_to_sequence

class Synthesizer(object):
    def __init__(self, config):
        self.wavernn = None
        self.config = config 
        self.use_cuda = config.use_cuda
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self.load_tts(self.config.tts_path, self.config.tts_file, self.config.tts_config, config.use_cuda)
        if self.config.wavernn_lib_path:
            self.load_wavernn(config.wavernn_lib_path, config.wavernn_path, config.wavernn_file, config.wavernn_config, config.use_cuda)

    def load_tts(self, model_path, model_file, model_config, use_cuda):
        tts_config = os.path.join(model_path, model_config)
        self.model_file = os.path.join(model_path, model_file)
        print(" > Loading TTS model ...")
        print(" | > model config: ", tts_config)
        print(" | > model file: ", model_file)
        self.tts_config = load_config(tts_config)
        self.use_phonemes = self.tts_config.use_phonemes
        self.ap = AudioProcessor(**self.tts_config.audio)
        if self.use_phonemes:
            self.input_size = len(phonemes)
            self.input_adapter = lambda sen: phoneme_to_sequence(sen, [self.tts_config.text_cleaner], self.tts_config.phoneme_language, self.tts_config.enable_eos_bos_chars)
        else:
            self.input_size = len(symbols)
            self.input_adapter = lambda sen: text_to_sequence(sen, [self.tts_config.text_cleaner])
        self.tts_model = setup_model(self.input_size, self.tts_config)
        # load model state
        if use_cuda:
            cp = torch.load(self.model_file)
        else:
            cp = torch.load(self.model_file, map_location=lambda storage, loc: storage)
        # load the model
        self.tts_model.load_state_dict(cp['model'])
        if use_cuda:
            self.tts_model.cuda()
        self.tts_model.eval()
        self.tts_model.decoder.max_decoder_steps = 3000

    def load_wavernn(self, lib_path, model_path, model_file, model_config, use_cuda):
        sys.path.append(lib_path) # set this if TTS is not installed globally
        from WaveRNN.models.wavernn import Model
        wavernn_config = os.path.join(model_path, model_config)
        model_file = os.path.join(model_path, model_file)
        print(" > Loading WaveRNN model ...")
        print(" | > model config: ", wavernn_config)
        print(" | > model file: ", model_file)
        self.wavernn_config = load_config(wavernn_config)
        self.wavernn = Model(
                rnn_dims=512,
                fc_dims=512,
                mode=self.wavernn_config.mode,
                pad=2,
                upsample_factors=self.wavernn_config.upsample_factors,  # set this depending on dataset
                feat_dims=80,
                compute_dims=128,
                res_out_dims=128,
                res_blocks=10,
                hop_length=self.ap.hop_length,
                sample_rate=self.ap.sample_rate,
            ).cuda()

        check = torch.load(model_file)
        self.wavernn.load_state_dict(check['model'])
        if use_cuda:
            self.wavernn.cuda()
        self.wavernn.eval()

    def save_wav(self, wav, path):
        # wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        wav = np.array(wav)
        self.ap.save_wav(wav, path)

    def split_into_sentences(self, text):
        text = " " + text + "  "
        text = text.replace("\n"," ")
        text = re.sub(prefixes,"\\1<prd>",text)
        text = re.sub(websites,"<prd>\\1",text)
        if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
        text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
        text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
        text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
        text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
        text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
        text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
        if "”" in text: text = text.replace(".”","”.")
        if "\"" in text: text = text.replace(".\"","\".")
        if "!" in text: text = text.replace("!\"","\"!")
        if "?" in text: text = text.replace("?\"","\"?")
        text = text.replace(".",".<stop>")
        text = text.replace("?","?<stop>")
        text = text.replace("!","!<stop>")
        text = text.replace("<prd>",".")
        sentences = text.split("<stop>")
        sentences = sentences[:-1]
        sentences = [s.strip() for s in sentences]
        return sentences

    def tts(self, text):
        wavs = []
        sens = self.split_into_sentences(text)
        if len(sens) == 0:
            sens = [text+'.']
        for sen in sens:
            if len(sen) < 3:
                continue
            sen = sen.strip()
            print(sen)

            seq = np.array(self.input_adapter(sen))
            text_hat = sequence_to_phoneme(seq)
            print(text_hat)

            chars_var = torch.from_numpy(seq).unsqueeze(0).long()

            if self.use_cuda:
                chars_var = chars_var.cuda()
            decoder_out, postnet_out, alignments, stop_tokens = self.tts_model.inference(
                chars_var)
            postnet_out = postnet_out[0].data.cpu().numpy()
            if self.tts_config.model == "Tacotron":
                wav = self.ap.inv_spectrogram(postnet_out.T)
            elif self.tts_config.model == "Tacotron2":
                if self.wavernn:
                    wav = self.wavernn.generate(torch.FloatTensor(postnet_out.T).unsqueeze(0).cuda(), batched=self.config.is_wavernn_batched, target=11000, overlap=550)
                else:
                    wav = self.ap.inv_mel_spectrogram(postnet_out.T)
            wavs += list(wav)
            wavs += [0] * 10000

        out = io.BytesIO()
        self.save_wav(wavs, out)
        return out
