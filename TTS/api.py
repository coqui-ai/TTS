from pathlib import Path
from TTS.utils.synthesizer import Synthesizer
from TTS.utils.manage import ModelManager


class TTS():
    """TODO: Add voice conversion and Capacitron support.
    """
    def __init__(self, model_name:str=None, progress_bar:bool=True, gpu=False):
        """🐸TTS python interface that allows to load and use the released models.

        Args:
            model_name (str, optional): Model name to load. You can list models by ```tts.models```. Defaults to None.
            progress_bar (bool, optional): Whether to pring a progress bar while downloading a model. Defaults to True.
            gpu (bool, optional): Enable/disable GPU. Some models might be too slow on CPU. Defaults to False.
        """
        self.manager = ModelManager(models_file=self.get_models_file_path(), progress_bar=progress_bar, verbose=False)
        self.synthesizer = None
        if model_name:
            self.load_model_by_name(model_name, gpu)

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

    def get_models_file_path(self):
        return Path(__file__).parent / ".models.json"

    def download_model_by_name(self, model_name:str):
        model_path, config_path, model_item = self.manager.download_model(model_name)
        if model_item["default_vocoder"] is None:
            return model_path, config_path, None, None
        vocoder_path, vocoder_config_path, _ = self.manager.download_model(model_item["default_vocoder"])
        return model_path, config_path, vocoder_path, vocoder_config_path

    def load_model_by_name(self, model_name:str, gpu:bool=False):
        model_path, config_path, vocoder_path, vocoder_config_path = self.download_model_by_name(model_name)
        # init synthesizer
        # None values are fetch from the model
        self.synthesizer = Synthesizer(
                                    tts_checkpoint = model_path,
                                    tts_config_path = config_path,
                                    tts_speakers_file = None,
                                    tts_languages_file= None,
                                    vocoder_checkpoint=vocoder_path,
                                    vocoder_config=vocoder_config_path,
                                    encoder_checkpoint = None,
                                    encoder_config=None,
                                    use_cuda=gpu,
                                )

    def _check_arguments(self, text:str, speaker:str=None, language:str=None):
        if self.is_multi_speaker and speaker is None:
            raise ValueError("Model is multi-speaker but no speaker is provided.")
        if self.is_multi_lingual and language is None:
            raise ValueError("Model is multi-lingual but no language is provided.")
        if not self.is_multi_speaker and speaker is not None:
            raise ValueError("Model is not multi-speaker but speaker is provided.")
        if not self.is_multi_lingual and language is not None:
            raise ValueError("Model is not multi-lingual but language is provided.")

    def tts(self, text:str, speaker:str=None, language:str=None):
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
        """
        self._check_arguments(text=text, speaker=speaker, language=language)

        wav = self.synthesizer.tts(
            text = text,
            speaker_name= speaker,
            language_name = language,
            speaker_wav=None,
            reference_wav=None,
            style_wav=None,
            style_text=None,
            reference_speaker_name=None,
        )
        return wav

    def tts_to_file(self, text:str, speaker:str=None, language:str=None, output_file_path:str="output.wav"):
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
            output_file_path (str, optional):
                Output file path. Defaults to "output.wav".
        """
        wav = self.tts(text=text, speaker=speaker, language=language)
        self.synthesizer.save_wav(wav=wav, path=output_file_path)



if __name__ == "__main__":
    tts = TTS()
    print(tts.models[0])

    # VITS
    # tts.load_model_by_name(tts.models[0])
    # print(tts.is_multi_speaker)
    # print(tts.is_multi_lingual)
    # print(tts.speakers)
    # print(tts.languages)

    # wav = tts.tts_to_file(text="Hello world!", speaker=tts.speakers[0], language=tts.languages[0])


    # tts.load_model_by_name(tts.models[1])
    # print(tts.is_multi_speaker)
    # print(tts.is_multi_lingual)
    # print(tts.speakers)
    # print(tts.languages)

    # # should raise an error
    # raised_error = False
    # try:
    #     wav = tts.tts_to_file(text="Hello world!", speaker="place_holder", language="place_holder")
    # except ValueError:
    #     raised_error = True
    # assert raised_error

    # wav = tts.tts_to_file(text="Hello world!")

    tts.load_model_by_name("tts_models/de/thorsten/tacotron2-DDC")
    print(tts.is_multi_speaker)
    print(tts.is_multi_lingual)
    print(tts.speakers)
    print(tts.languages)

    # should raise an error
    raised_error = False
    try:
        wav = tts.tts_to_file(text="Hello world!", speaker="place_holder", language="place_holder")
    except ValueError:
        raised_error = True
    assert raised_error

    wav = tts.tts_to_file(text="Ich heiße David und ich bin ein deutscher Sprecher.")
