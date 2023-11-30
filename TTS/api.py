import tempfile
import warnings
from pathlib import Path
from typing import Union

import numpy as np
from torch import nn

from TTS.cs_api import CS_API
from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer
from TTS.config import load_config


class TTS(nn.Module):
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_name: str = "",
        model_path: str = None,
        config_path: str = None,
        vocoder_path: str = None,
        vocoder_config_path: str = None,
        progress_bar: bool = True,
        cs_api_model: str = "XTTS",
        gpu=False,
    ):
        """🐸TTS python interface that allows to load and use the released models.

        Example with a multi-speaker model:
            >>> from TTS.api import TTS
            >>> tts = TTS(TTS.list_models()[0])
            >>> wav = tts.tts("This is a test! This is also a test!!", speaker=tts.speakers[0], language=tts.languages[0])
            >>> tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0], file_path="output.wav")

        Example with a single-speaker model:
            >>> tts = TTS(model_name="tts_models/de/thorsten/tacotron2-DDC", progress_bar=False, gpu=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example loading a model from a path:
            >>> tts = TTS(model_path="/path/to/checkpoint_100000.pth", config_path="/path/to/config.json", progress_bar=False, gpu=False)
            >>> tts.tts_to_file(text="Ich bin eine Testnachricht.", file_path="output.wav")

        Example voice cloning with YourTTS in English, French and Portuguese:
            >>> tts = TTS(model_name="tts_models/multilingual/multi-dataset/your_tts", progress_bar=False, gpu=True)
            >>> tts.tts_to_file("This is voice cloning.", speaker_wav="my/cloning/audio.wav", language="en", file_path="thisisit.wav")
            >>> tts.tts_to_file("C'est le clonage de la voix.", speaker_wav="my/cloning/audio.wav", language="fr", file_path="thisisit.wav")
            >>> tts.tts_to_file("Isso é clonagem de voz.", speaker_wav="my/cloning/audio.wav", language="pt", file_path="thisisit.wav")

        Example Fairseq TTS models (uses ISO language codes in https://dl.fbaipublicfiles.com/mms/tts/all-tts-languages.html):
            >>> tts = TTS(model_name="tts_models/eng/fairseq/vits", progress_bar=False, gpu=True)
            >>> tts.tts_to_file("This is a test.", file_path="output.wav")

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to None.
            config_path (str, optional): Path to the model config. Defaults to None.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
            progress_bar (bool, optional): Whether to pring a progress bar while downloading a model. Defaults to True.
            cs_api_model (str, optional): Name of the model to use for the Coqui Studio API. Available models are
                "XTTS", "V1". You can also use `TTS.cs_api.CS_API" for more control.
                Defaults to "XTTS".
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        super().__init__()
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar, verbose=False)
        self.config = load_config(config_path) if config_path else None
        self.synthesizer = None
        self.voice_converter = None
        self.csapi = None
        self.cs_api_model = cs_api_model
        self.model_name = ""
        if gpu:
            warnings.warn("`gpu` will be deprecated. Please use `tts.to(device)` instead.")

        if model_name is not None and len(model_name) > 0:
            if "tts_models" in model_name or "coqui_studio" in model_name:
                self.load_tts_model_by_name(model_name, gpu)
            elif "voice_conversion_models" in model_name:
                self.load_vc_model_by_name(model_name, gpu)
            else:
                self.load_model_by_name(model_name, gpu)

        if model_path:
            self.load_tts_model_by_path(
                model_path, config_path, vocoder_path=vocoder_path, vocoder_config=vocoder_config_path, gpu=gpu
            )

    @property
    def models(self):
        return self.manager.list_tts_models()

    @property
    def is_multi_speaker(self):
        if hasattr(self.synthesizer.tts_model, "speaker_manager") and self.synthesizer.tts_model.speaker_manager:
            return self.synthesizer.tts_model.speaker_manager.num_speakers > 1
        return False

    @property
    def is_coqui_studio(self):
        if self.model_name is None:
            return False
        return "coqui_studio" in self.model_name

    @property
    def is_multi_lingual(self):
        # Not sure what sets this to None, but applied a fix to prevent crashing.
        if (
            isinstance(self.model_name, str)
            and "xtts" in self.model_name
            or self.config
            and ("xtts" in self.config.model or len(self.config.languages) > 1)
        ):
            return True
        if hasattr(self.synthesizer.tts_model, "language_manager") and self.synthesizer.tts_model.language_manager:
            return self.synthesizer.tts_model.language_manager.num_languages > 1
        return False

    @property
    def speakers(self):
        if not self.is_multi_speaker:
            return None
        return self.synthesizer.tts_model.speaker_manager.speaker_names

    @property
    def languages(self):
        if not self.is_multi_lingual:
            return None
        return self.synthesizer.tts_model.language_manager.language_names

    @staticmethod
    def get_models_file_path():
        return Path(__file__).parent / ".models.json"

    def list_models(self):
        try:
            csapi = CS_API(model=self.cs_api_model)
            models = csapi.list_speakers_as_tts_models()
        except ValueError as e:
            print(e)
            models = []
        manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False)
        return manager.list_tts_models() + models

    def download_model_by_name(self, model_name: str):
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if "fairseq" in model_name or (model_item is not None and isinstance(model_item["model_url"], list)):
            # return model directory if there are multiple files
            # we assume that the model knows how to load itself
            return None, None, None, None, model_path
        if model_item.get("default_vocoder") is None:
            return model_path, config_path, None, None, None
        vocoder_path, vocoder_config_path, _ = self.manager.download_model(model_item["default_vocoder"])
        return model_path, config_path, vocoder_path, vocoder_config_path, None

    def load_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.load_tts_model_by_name(model_name, gpu)

    def load_vc_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the voice conversion models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.model_name = model_name
        model_path, config_path, _, _, _ = self.download_model_by_name(model_name)
        self.voice_converter = Synthesizer(vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

    def load_tts_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.

        TODO: Add tests
        """
        self.synthesizer = None
        self.csapi = None
        self.model_name = model_name

        if "coqui_studio" in model_name:
            self.csapi = CS_API()
        else:
            model_path, config_path, vocoder_path, vocoder_config_path, model_dir = self.download_model_by_name(
                model_name
            )

            # init synthesizer
            # None values are fetch from the model
            self.synthesizer = Synthesizer(
                tts_checkpoint=model_path,
                tts_config_path=config_path,
                tts_speakers_file=None,
                tts_languages_file=None,
                vocoder_checkpoint=vocoder_path,
                vocoder_config=vocoder_config_path,
                encoder_checkpoint=None,
                encoder_config=None,
                model_dir=model_dir,
                use_cuda=gpu,
            )

    def load_tts_model_by_path(
        self, model_path: str, config_path: str, vocoder_path: str = None, vocoder_config: str = None, gpu: bool = False
    ):
        """Load a model from a path.

        Args:
            model_path (str): Path to the model checkpoint.
            config_path (str): Path to the model config.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config (str, optional): Path to the vocoder config. Defaults to None.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """

        self.synthesizer = Synthesizer(
            tts_checkpoint=model_path,
            tts_config_path=config_path,
            tts_speakers_file=None,
            tts_languages_file=None,
            vocoder_checkpoint=vocoder_path,
            vocoder_config=vocoder_config,
            encoder_checkpoint=None,
            encoder_config=None,
            use_cuda=gpu,
        )

    def _check_arguments(
        self,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
        **kwargs,
    ) -> None:
        """Check if the arguments are valid for the model."""
        if not self.is_coqui_studio:
            # check for the coqui tts models
            if self.is_multi_speaker and (speaker is None and speaker_wav is None):
                raise ValueError("Model is multi-speaker but no `speaker` is provided.")
            if self.is_multi_lingual and language is None:
                raise ValueError("Model is multi-lingual but no `language` is provided.")
            if not self.is_multi_speaker and speaker is not None and "voice_dir" not in kwargs:
                raise ValueError("Model is not multi-speaker but `speaker` is provided.")
            if not self.is_multi_lingual and language is not None:
                raise ValueError("Model is not multi-lingual but `language` is provided.")
            if not emotion is None and not speed is None:
                raise ValueError("Emotion and speed can only be used with Coqui Studio models.")
        else:
            if emotion is None:
                emotion = "Neutral"
            if speed is None:
                speed = 1.0
            # check for the studio models
            if speaker_wav is not None:
                raise ValueError("Coqui Studio models do not support `speaker_wav` argument.")
            if speaker is not None:
                raise ValueError("Coqui Studio models do not support `speaker` argument.")
            if language is not None and language != "en":
                raise ValueError("Coqui Studio models currently support only `language=en` argument.")
            if emotion not in ["Neutral", "Happy", "Sad", "Angry", "Dull"]:
                raise ValueError(f"Emotion - `{emotion}` - must be one of `Neutral`, `Happy`, `Sad`, `Angry`, `Dull`.")

    def tts_coqui_studio(
        self,
        text: str,
        speaker_name: str = None,
        language: str = None,
        emotion: str = None,
        speed: float = 1.0,
        pipe_out=None,
        file_path: str = None,
    ) -> Union[np.ndarray, str]:
        """Convert text to speech using Coqui Studio models. Use `CS_API` class if you are only interested in the API.

        Args:
            text (str):
                Input text to synthesize.
            speaker_name (str, optional):
                Speaker name from Coqui Studio. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            emotion (str, optional):
                Emotion of the speaker. One of "Neutral", "Happy", "Sad", "Angry", "Dull". Emotions are only available
                with "V1" model. Defaults to None.
            speed (float, optional):
                Speed of the speech. Defaults to 1.0.
            pipe_out (BytesIO, optional):
                Flag to stdout the generated TTS wav file for shell pipe.
            file_path (str, optional):
                Path to save the output file. When None it returns the `np.ndarray` of waveform. Defaults to None.

        Returns:
            Union[np.ndarray, str]: Waveform of the synthesized speech or path to the output file.
        """
        speaker_name = self.model_name.split("/")[2]
        if file_path is not None:
            return self.csapi.tts_to_file(
                text=text,
                speaker_name=speaker_name,
                language=language,
                speed=speed,
                pipe_out=pipe_out,
                emotion=emotion,
                file_path=file_path,
            )[0]
        return self.csapi.tts(text=text, speaker_name=speaker_name, language=language, speed=speed, emotion=emotion)[0]

    def tts(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = None,
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. If None, Studio models use "Neutral". Defaults to None.
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0 and 2.0. If None, Studio models use 1.0.
                Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        self._check_arguments(
            speaker=speaker, language=language, speaker_wav=speaker_wav, emotion=emotion, speed=speed, **kwargs
        )
        if self.csapi is not None:
            return self.tts_coqui_studio(
                text=text, speaker_name=speaker, language=language, emotion=emotion, speed=speed
            )
        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
            split_sentences=split_sentences,
            **kwargs,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        emotion: str = None,
        speed: float = 1.0,
        pipe_out=None,
        file_path: str = "output.wav",
        split_sentences: bool = True,
        **kwargs,
    ):
        """Convert text to speech.

        Args:
            text (str):
                Input text to synthesize.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            emotion (str, optional):
                Emotion to use for 🐸Coqui Studio models. Defaults to "Neutral".
            speed (float, optional):
                Speed factor to use for 🐸Coqui Studio models, between 0.0 and 2.0. Defaults to None.
            pipe_out (BytesIO, optional):
                Flag to stdout the generated TTS wav file for shell pipe.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
            kwargs (dict, optional):
                Additional arguments for the model.
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav, **kwargs)

        if self.csapi is not None:
            return self.tts_coqui_studio(
                text=text,
                speaker_name=speaker,
                language=language,
                emotion=emotion,
                speed=speed,
                file_path=file_path,
                pipe_out=pipe_out,
            )
        wav = self.tts(
            text=text,
            speaker=speaker,
            language=language,
            speaker_wav=speaker_wav,
            split_sentences=split_sentences,
            **kwargs,
        )
        self.synthesizer.save_wav(wav=wav, path=file_path, pipe_out=pipe_out)
        return file_path

    def voice_conversion(
        self,
        source_wav: str,
        target_wav: str,
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker.

        Args:``
            source_wav (str):
                Path to the source wav file.
            target_wav (str):`
                Path to the target wav file.
        """
        wav = self.voice_converter.voice_conversion(source_wav=source_wav, target_wav=target_wav)
        return wav

    def voice_conversion_to_file(
        self,
        source_wav: str,
        target_wav: str,
        file_path: str = "output.wav",
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker.

        Args:
            source_wav (str):
                Path to the source wav file.
            target_wav (str):
                Path to the target wav file.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
        """
        wav = self.voice_conversion(source_wav=source_wav, target_wav=target_wav)
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
        return file_path

    def tts_with_vc(
        self,
        text: str,
        language: str = None,
        speaker_wav: str = None,
        speaker: str = None,
        split_sentences: bool = True,
    ):
        """Convert text to speech with voice conversion.

        It combines tts with voice conversion to fake voice cloning.

        - Convert text to speech with tts.
        - Convert the output wav to target speaker with voice conversion.

        Args:
            text (str):
                Input text to synthesize.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            # Lazy code... save it to a temp file to resample it while reading it for VC
            self.tts_to_file(
                text=text, speaker=speaker, language=language, file_path=fp.name, split_sentences=split_sentences
            )
        if self.voice_converter is None:
            self.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
        wav = self.voice_converter.voice_conversion(source_wav=fp.name, target_wav=speaker_wav)
        return wav

    def tts_with_vc_to_file(
        self,
        text: str,
        language: str = None,
        speaker_wav: str = None,
        file_path: str = "output.wav",
        speaker: str = None,
        split_sentences: bool = True,
    ):
        """Convert text to speech with voice conversion and save to file.

        Check `tts_with_vc` for more details.

        Args:
            text (str):
                Input text to synthesize.
            language (str, optional):
                Language code for multi-lingual models. You can check whether loaded model is multi-lingual
                `tts.is_multi_lingual` and list available languages by `tts.languages`. Defaults to None.
            speaker_wav (str, optional):
                Path to a reference wav file to use for voice cloning with supporting models like YourTTS.
                Defaults to None.
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
            speaker (str, optional):
                Speaker name for multi-speaker. You can check whether loaded model is multi-speaker by
                `tts.is_multi_speaker` and list speakers by `tts.speakers`. Defaults to None.
            split_sentences (bool, optional):
                Split text into sentences, synthesize them separately and concatenate the file audio.
                Setting it False uses more VRAM and possibly hit model specific text length or VRAM limits. Only
                applicable to the 🐸TTS models. Defaults to True.
        """
        wav = self.tts_with_vc(
            text=text, language=language, speaker_wav=speaker_wav, speaker=speaker, split_sentences=split_sentences
        )
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
