import time

import numpy as np
import torch
import pysbd

from TTS.utils.audio import AudioProcessor
from TTS.utils.io import load_config
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.speakers import load_speaker_mapping
from TTS.vocoder.utils.generic_utils import setup_generator, interpolate_vocoder_input
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
from TTS.tts.utils.synthesis import synthesis, trim_silence
from TTS.tts.utils.text import make_symbols, phonemes, symbols


class Synthesizer(object):
    def __init__(self, tts_checkpoint, tts_config, vocoder_checkpoint=None, vocoder_config=None, use_cuda=False):
        """Encapsulation of tts and vocoder models for inference.

        TODO: handle multi-speaker and GST inference.

        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config = tts_config
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.use_cuda = use_cuda
        self.wavernn = None
        self.vocoder_model = None
        self.num_speakers = 0
        self.tts_speakers = None
        self.speaker_embedding_dim = None
        self.seg = self.get_segmenter("en")
        self.use_cuda = use_cuda
        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."
        self.load_tts(tts_checkpoint, tts_config,
                      use_cuda)
        self.output_sample_rate = self.tts_config.audio['sample_rate']
        if vocoder_checkpoint:
            self.load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio['sample_rate']

    @staticmethod
    def get_segmenter(lang):
        return pysbd.Segmenter(language=lang, clean=True)

    def load_speakers(self):
        # load speakers
        if self.model_config.use_speaker_embedding is not None:
            self.tts_speakers = load_speaker_mapping(self.tts_config.tts_speakers_json)
            self.num_speakers = len(self.tts_speakers)
        else:
            self.num_speakers = 0
        # set external speaker embedding
        if self.tts_config.use_external_speaker_embedding_file:
            speaker_embedding = self.tts_speakers[list(self.tts_speakers.keys())[0]]['embedding']
            self.speaker_embedding_dim = len(speaker_embedding)

    def init_speaker(self, speaker_idx):
        # load speakers
        speaker_embedding = None
        if hasattr(self, 'tts_speakers') and speaker_idx is not None:
            assert speaker_idx < len(self.tts_speakers), f" [!] speaker_idx is out of the range. {speaker_idx} vs {len(self.tts_speakers)}"
            if self.tts_config.use_external_speaker_embedding_file:
                speaker_embedding = self.tts_speakers[speaker_idx]['embedding']
        return speaker_embedding

    def load_tts(self, tts_checkpoint, tts_config, use_cuda):
        # pylint: disable=global-statement

        global symbols, phonemes

        self.tts_config = load_config(tts_config)
        self.use_phonemes = self.tts_config.use_phonemes
        self.ap = AudioProcessor(verbose=False, **self.tts_config.audio)

        if 'characters' in self.tts_config.keys():
            symbols, phonemes = make_symbols(**self.tts_config.characters)

        if self.use_phonemes:
            self.input_size = len(phonemes)
        else:
            self.input_size = len(symbols)

        self.tts_model = setup_model(self.input_size, num_speakers=self.num_speakers, c=self.tts_config)
        self.tts_model.load_checkpoint(tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()

    def load_vocoder(self, model_file, model_config, use_cuda):
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config['audio'])
        self.vocoder_model = setup_generator(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def save_wav(self, wav, path):
        wav = np.array(wav)
        self.ap.save_wav(wav, path, self.output_sample_rate)

    def split_into_sentences(self, text):
        return self.seg.segment(text)

    def tts(self, text, speaker_idx=None):
        start_time = time.time()
        wavs = []
        sens = self.split_into_sentences(text)
        print(" > Text splitted to sentences.")
        print(sens)

        speaker_embedding = self.init_speaker(speaker_idx)
        use_gl = self.vocoder_model is None

        for sen in sens:
            # synthesize voice
            waveform, _, _, mel_postnet_spec, _, _ = synthesis(
                self.tts_model,
                sen,
                self.tts_config,
                self.use_cuda,
                self.ap,
                speaker_idx,
                None,
                False,
                self.tts_config.enable_eos_bos_chars,
                use_gl,
                speaker_embedding=speaker_embedding)
            if not use_gl:
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [1, self.vocoder_config['audio']['sample_rate'] / self.ap.sample_rate]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(scale_factor, vocoder_input)
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(0)  # pylint: disable=not-callable
                # run vocoder model
                # [1, T, C]
                waveform = self.vocoder_model.inference(vocoder_input.to(device_type))
            if self.use_cuda and not use_gl:
                waveform = waveform.cpu()
            if not use_gl:
                waveform = waveform.numpy()
            waveform = waveform.squeeze()

            # trim silence
            waveform = trim_silence(waveform, self.ap)

            wavs += list(waveform)
            wavs += [0] * 10000

        # compute stats
        process_time = time.time() - start_time
        audio_time = len(wavs) / self.tts_config.audio['sample_rate']
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs
