import io
import os
import librosa
import torch
import scipy
import numpy as np
import soundfile as sf
from TTS.utils.text import text_to_sequence
from TTS.utils.generic_utils import load_config
from TTS.utils.audio import AudioProcessor
from TTS.models.tacotron import Tacotron
from matplotlib import pylab as plt


class Synthesizer(object):
    def load_model(self, model_path, model_name, model_config, use_cuda):
        model_config = os.path.join(model_path, model_config)
        self.model_file = os.path.join(model_path, model_name)
        print(" > Loading model ...")
        print(" | > model config: ", model_config)
        print(" | > model file: ", self.model_file)
        config = load_config(model_config)
        self.config = config
        self.use_cuda = use_cuda
        self.model = Tacotron(config.embedding_size, config.num_freq,
                              config.num_mels, config.r)
        self.ap = AudioProcessor(
            config.sample_rate,
            config.num_mels,
            config.min_level_db,
            config.frame_shift_ms,
            config.frame_length_ms,
            config.preemphasis,
            config.ref_level_db,
            config.num_freq,
            config.power,
            griffin_lim_iters=60)
        # load model state
        if use_cuda:
            cp = torch.load(self.model_file)
        else:
            cp = torch.load(
                self.model_file, map_location=lambda storage, loc: storage)
        # load the model
        self.model.load_state_dict(cp['model'])
        if use_cuda:
            self.model.cuda()
        self.model.eval()

    def save_wav(self, wav, path):
        wav *= 32767 / max(1e-8, np.max(np.abs(wav)))
        librosa.output.write_wav(path, wav.astype(np.int16),
                                 self.config.sample_rate)

    def tts(self, text):
        text_cleaner = [self.config.text_cleaner]
        wavs = []
        for sen in text.split('.'):
            if len(sen) < 3:
                continue
            sen = sen.strip()
            sen += '.'
            print(sen)
            sen = sen.strip()
            seq = np.array(text_to_sequence(text, text_cleaner))
            chars_var = torch.from_numpy(seq).unsqueeze(0).long()
            if self.use_cuda:
                chars_var = chars_var.cuda()
            mel_out, linear_out, alignments, stop_tokens = self.model.forward(
                chars_var)
            linear_out = linear_out[0].data.cpu().numpy()
            wav = self.ap.inv_spectrogram(linear_out.T)
            # wav = wav[:self.ap.find_endpoint(wav)]
            out = io.BytesIO()
            wavs.append(wav)
            wavs.append(np.zeros(10000))
        self.save_wav(wav, out)
        return out
