import time
from typing import List

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
    def __init__(
        self,
        tts_checkpoint: str,
        tts_config_path: str,
        vocoder_checkpoint: str = "",
        vocoder_config: str = "",
        use_cuda: bool = False,
    ) -> None:
        """Encapsulation of tts and vocoder models for inference.

        Args:
            tts_checkpoint (str): path to the tts model file.
            tts_config_path (str): path to the tts config file.
            vocoder_checkpoint (str, optional): path to the vocoder model file. Defaults to None.
            vocoder_config (str, optional): path to the vocoder config file. Defaults to None.
            use_cuda (bool, optional): enable/disable cuda. Defaults to False.
        """
        self.tts_checkpoint = tts_checkpoint
        self.tts_config_path = tts_config_path
        self.vocoder_checkpoint = vocoder_checkpoint
        self.vocoder_config = vocoder_config
        self.use_cuda = use_cuda
        self.vocoder_model = None
        self.num_speakers = 0
        self.tts_speakers = {}
        self.speaker_embedding_dim = 0
        self.seg = self._get_segmenter("en")
        self.use_cuda = use_cuda

        if self.use_cuda:
            assert torch.cuda.is_available(), "CUDA is not availabe on this machine."



        self._load_tts(tts_checkpoint, tts_config_path, use_cuda)
        self.output_sample_rate = self.tts_config.audio["sample_rate"]
        if vocoder_checkpoint:
            self._load_vocoder(vocoder_checkpoint, vocoder_config, use_cuda)
            self.output_sample_rate = self.vocoder_config.audio["sample_rate"]
        

    @staticmethod
    def _get_segmenter(lang: str):
        return pysbd.Segmenter(language=lang, clean=True)


    def _load_speakers(self) -> None:
        print("Loading speakers ...")
        self.tts_speakers = load_speaker_mapping(self.tts_config.external_speaker_embedding_file)
        self.num_speakers = len(self.tts_speakers)
        self.speaker_embedding_dim = len(self.tts_speakers[list(self.tts_speakers.keys())[0]][
            "embedding"
        ])

    def _load_speaker_embedding(self, speaker_json_key: str = ""):

        speaker_embedding = None
        
        if self.tts_config.get("use_external_speaker_embedding_file") and not speaker_json_key:
            raise ValueError("While 'use_external_speaker_embedding_file', you must pass a 'speaker_json_key'")
        
        if speaker_json_key != "":
            assert self.tts_speakers
            assert speaker_json_key in self.tts_speakers, f"speaker_json_key is not in self.tts_speakers keys : '{speaker_idx}'"
            speaker_embedding = self.tts_speakers[speaker_json_key]["embedding"]

        return speaker_embedding

    def _load_tts(
        self, tts_checkpoint: str, tts_config_path: str, use_cuda: bool
    ) -> None:
        # pylint: disable=global-statement

        global symbols, phonemes

        self.tts_config = load_config(tts_config_path)
        self.use_phonemes = self.tts_config.use_phonemes
        self.ap = AudioProcessor(verbose=False, **self.tts_config.audio)

        if "characters" in self.tts_config.keys():
            symbols, phonemes = make_symbols(**self.tts_config.characters)

        if self.use_phonemes:
            self.input_size = len(phonemes)
        else:
            self.input_size = len(symbols)

        if self.tts_config.use_speaker_embedding is True:
            self._load_speakers()

        self.tts_model = setup_model(
            self.input_size, 
            num_speakers=self.num_speakers, 
            c=self.tts_config, 
            speaker_embedding_dim=self.speaker_embedding_dim)

        self.tts_model.load_checkpoint(self.tts_config, tts_checkpoint, eval=True)
        if use_cuda:
            self.tts_model.cuda()



    def _load_vocoder(self, model_file: str, model_config: str, use_cuda: bool) -> None:
        self.vocoder_config = load_config(model_config)
        self.vocoder_ap = AudioProcessor(verbose=False, **self.vocoder_config["audio"])
        self.vocoder_model = setup_generator(self.vocoder_config)
        self.vocoder_model.load_checkpoint(self.vocoder_config, model_file, eval=True)
        if use_cuda:
            self.vocoder_model.cuda()

    def _split_into_sentences(self, text) -> List[str]:
        return self.seg.segment(text)

    def save_wav(self, wav: List[int], path: str) -> None:
        wav = np.array(wav)
        self.ap.save_wav(wav, path, self.output_sample_rate)

    def tts(self, text: str, speaker_json_key: str = "", style_wav = None) -> List[int]:
        start_time = time.time()
        wavs = []
        sens = self._split_into_sentences(text)
        print(" > Text splitted to sentences.")
        print(sens)
        speaker_embedding = self._load_speaker_embedding(speaker_json_key)
        
        use_gl = self.vocoder_model is None

        for sen in sens:
            # synthesize voice
            waveform, _, _, mel_postnet_spec, _, _ = synthesis(
                model=self.tts_model,
                text=sen,
                CONFIG=self.tts_config,
                use_cuda=self.use_cuda,
                ap=self.ap,
                speaker_id=None,
                style_wav=style_wav,
                truncated=False,
                enable_eos_bos_chars=self.tts_config.enable_eos_bos_chars,
                use_griffin_lim=use_gl,
                speaker_embedding=speaker_embedding,
            )
            if not use_gl:
                # denormalize tts output based on tts audio config
                mel_postnet_spec = self.ap.denormalize(mel_postnet_spec.T).T
                device_type = "cuda" if self.use_cuda else "cpu"
                # renormalize spectrogram based on vocoder config
                vocoder_input = self.vocoder_ap.normalize(mel_postnet_spec.T)
                # compute scale factor for possible sample rate mismatch
                scale_factor = [
                    1,
                    self.vocoder_config["audio"]["sample_rate"] / self.ap.sample_rate,
                ]
                if scale_factor[1] != 1:
                    print(" > interpolating tts model output.")
                    vocoder_input = interpolate_vocoder_input(
                        scale_factor, vocoder_input
                    )
                else:
                    vocoder_input = torch.tensor(vocoder_input).unsqueeze(
                        0
                    )  # pylint: disable=not-callable
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
        audio_time = len(wavs) / self.tts_config.audio["sample_rate"]
        print(f" > Processing time: {process_time}")
        print(f" > Real-time factor: {process_time / audio_time}")
        return wavs
