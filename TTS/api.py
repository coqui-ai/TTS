import tempfile
from pathlib import Path

from TTS.utils.audio.numpy_transforms import save_wav
from TTS.utils.manage import ModelManager
from TTS.utils.synthesizer import Synthesizer


class TTS:
    """TODO: Add voice conversion and Capacitron support."""

    def __init__(
        self,
        model_name: str = None,
        model_path: str = None,
        config_path: str = None,
        vocoder_path: str = None,
        vocoder_config_path: str = None,
        progress_bar: bool = True,
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

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            model_path (str, optional): Path to the model checkpoint. Defaults to None.
            config_path (str, optional): Path to the model config. Defaults to None.
            vocoder_path (str, optional): Path to the vocoder checkpoint. Defaults to None.
            vocoder_config_path (str, optional): Path to the vocoder config. Defaults to None.
            progress_bar (bool, optional): Whether to pring a progress bar while downloading a model. Defaults to True.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar, verbose=False)

        self.synthesizer = None
        self.voice_converter = None

        if model_name:
            self.load_tts_model_by_name(model_name, gpu)
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
    def is_multi_lingual(self):
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

    @staticmethod
    def list_models():
        manager = ModelManager(models_file=TTS.get_models_file_path(), progress_bar=False, verbose=False)
        return manager.list_tts_models()

    def download_model_by_name(self, model_name: str):
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if model_item.get("default_vocoder") is None:
            return model_path, config_path, None, None
        vocoder_path, vocoder_config_path, _ = self.manager.download_model(model_item["default_vocoder"])
        return model_path, config_path, vocoder_path, vocoder_config_path

    def load_vc_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of the voice conversion models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        model_path, config_path, _, _ = self.download_model_by_name(model_name)
        self.voice_converter = Synthesizer(vc_checkpoint=model_path, vc_config=config_path, use_cuda=gpu)

    def load_tts_model_by_name(self, model_name: str, gpu: bool = False):
        """Load one of 🐸TTS models by name.

        Args:
            model_name (str): Model name to load. You can list models by ```tts.models```.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.

        TODO: Add tests
        """

        model_path, config_path, vocoder_path, vocoder_config_path = self.download_model_by_name(model_name)

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

    def _check_arguments(self, speaker: str = None, language: str = None, speaker_wav: str = None):
        if self.is_multi_speaker and (speaker is None and speaker_wav is None):
            raise ValueError("Model is multi-speaker but no speaker is provided.")
        if self.is_multi_lingual and language is None:
            raise ValueError("Model is multi-lingual but no language is provided.")
        if not self.is_multi_speaker and speaker is not None:
            raise ValueError("Model is not multi-speaker but speaker is provided.")
        if not self.is_multi_lingual and language is not None:
            raise ValueError("Model is not multi-lingual but language is provided.")

    def tts(self, text: str, speaker: str = None, language: str = None, speaker_wav: str = None):
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
        """
        self._check_arguments(speaker=speaker, language=language, speaker_wav=speaker_wav)

        wav = self.synthesizer.tts(
            text=text,
            speaker_name=speaker,
            language_name=language,
            speaker_wav=speaker_wav,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
        )
        return wav

    def tts_to_file(
        self,
        text: str,
        speaker: str = None,
        language: str = None,
        speaker_wav: str = None,
        file_path: str = "output.wav",
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
            file_path (str, optional):
                Output file path. Defaults to "output.wav".
        """
        wav = self.tts(text=text, speaker=speaker, language=language, speaker_wav=speaker_wav)
        self.synthesizer.save_wav(wav=wav, path=file_path)

    def voice_conversion(
        self,
        sourve_wav: str,
        target_wav: str,
    ):
        """Voice conversion with FreeVC. Convert source wav to target speaker.

        Args:
            source_wav (str):
                Path to the source wav file.
            target_wav (str):
                Path to the target wav file.
        """
        wav = self.synthesizer.voice_conversion(source_wav=sourve_wav, target_wav=target_wav)
        return wav

    def tts_with_vc(self, text: str, language: str = None, speaker_wav: str = None):
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
        """
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            # Lazy code... save it to a temp file to resample it while reading it for VC
            self.tts_to_file(text=text, speaker=None, language=language, file_path=fp.name)
        if self.voice_converter is None:
            self.load_vc_model_by_name("voice_conversion_models/multilingual/vctk/freevc24")
        wav = self.voice_converter.voice_conversion(source_wav=fp.name, target_wav=speaker_wav)
        return wav

    def tts_with_vc_to_file(
        self, text: str, language: str = None, speaker_wav: str = None, file_path: str = "output.wav"
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
        """
        wav = self.tts_with_vc(text=text, language=language, speaker_wav=speaker_wav)
        save_wav(wav=wav, path=file_path, sample_rate=self.voice_converter.vc_config.audio.output_sample_rate)
