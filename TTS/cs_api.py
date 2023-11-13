import http.client
import json
import os
import tempfile
import urllib.request
from typing import Tuple

import numpy as np
import requests
from scipy.io import wavfile

from TTS.utils.audio.numpy_transforms import save_wav


class Speaker(object):
    """Convert dict to object."""

    def __init__(self, d, is_voice=False):
        self.is_voice = is_voice
        for k, v in d.items():
            if isinstance(k, (list, tuple)):
                setattr(self, k, [Speaker(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Speaker(v) if isinstance(v, dict) else v)

    def __repr__(self):
        return str(self.__dict__)


class CS_API:
    """🐸Coqui Studio API Wrapper.

    🐸Coqui Studio is the most advanced voice generation platform. You can generate new voices by voice cloning, voice
    interpolation, or our unique prompt to voice technology. It also provides a set of built-in voices with different
    characteristics. You can use these voices to generate new audio files or use them in your applications.
    You can use all the built-in and your own 🐸Coqui Studio speakers with this API with an API token.
    You can signup to 🐸Coqui Studio from https://app.coqui.ai/auth/signup and get an API token from
    https://app.coqui.ai/account. We can either enter the token as an environment variable as
    `export COQUI_STUDIO_TOKEN=<token>` or pass it as `CS_API(api_token=<toke>)`.
    Visit https://app.coqui.ai/api for more information.


    Args:
        api_token (str): 🐸Coqui Studio API token. If not provided, it will be read from the environment variable
            `COQUI_STUDIO_TOKEN`.
        model (str): 🐸Coqui Studio model. It can be either `V1`, `XTTS`. Default is `XTTS`.


    Example listing all available speakers:
        >>> from TTS.api import CS_API
        >>> tts = CS_API()
        >>> tts.speakers

    Example listing all emotions:
        >>> # emotions are only available for `V1` model
        >>> from TTS.api import CS_API
        >>> tts = CS_API(model="V1")
        >>> tts.emotions

    Example with a built-in 🐸 speaker:
        >>> from TTS.api import CS_API
        >>> tts = CS_API()
        >>> wav, sr = api.tts("Hello world", speaker_name=tts.speakers[0].name)
        >>> filepath = tts.tts_to_file(text="Hello world!", speaker_name=tts.speakers[0].name, file_path="output.wav")

    Example with multi-language model:
        >>> from TTS.api import CS_API
        >>> tts = CS_API(model="XTTS")
        >>> wav, sr = api.tts("Hello world", speaker_name=tts.speakers[0].name, language="en")
    """

    MODEL_ENDPOINTS = {
        "V1": {
            "list_speakers": "https://app.coqui.ai/api/v2/speakers",
            "synthesize": "https://app.coqui.ai/api/v2/samples",
            "list_voices": "https://app.coqui.ai/api/v2/voices",
        },
        "XTTS": {
            "list_speakers": "https://app.coqui.ai/api/v2/speakers",
            "synthesize": "https://app.coqui.ai/api/v2/samples/xtts/render/",
            "list_voices": "https://app.coqui.ai/api/v2/voices/xtts",
        },
    }

    SUPPORTED_LANGUAGES = ["en", "es", "de", "fr", "it", "pt", "pl", "tr", "ru", "nl", "cs", "ar", "zh-cn", "ja"]

    def __init__(self, api_token=None, model="XTTS"):
        self.api_token = api_token
        self.model = model
        self.headers = None
        self._speakers = None
        self._check_token()

    @staticmethod
    def ping_api():
        URL = "https://coqui.gateway.scarf.sh/tts/api"
        _ = requests.get(URL)

    @property
    def speakers(self):
        if self._speakers is None:
            self._speakers = self.list_all_speakers()
        return self._speakers

    @property
    def emotions(self):
        """Return a list of available emotions.

        TODO: Get this from the API endpoint.
        """
        if self.model == "V1":
            return ["Neutral", "Happy", "Sad", "Angry", "Dull"]
        else:
            raise ValueError(f"❗ Emotions are not available for {self.model}.")

    def _check_token(self):
        if self.api_token is None:
            self.api_token = os.environ.get("COQUI_STUDIO_TOKEN")
            self.headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_token}"}
        if not self.api_token:
            raise ValueError(
                "No API token found for 🐸Coqui Studio voices - https://coqui.ai \n"
                "Visit 🔗https://app.coqui.ai/account to get one.\n"
                "Set it as an environment variable `export COQUI_STUDIO_TOKEN=<token>`\n"
                ""
            )

    def list_all_speakers(self):
        """Return both built-in Coqui Studio speakers and custom voices created by the user."""
        return self.list_speakers() + self.list_voices()

    def list_speakers(self):
        """List built-in Coqui Studio speakers."""
        self._check_token()
        conn = http.client.HTTPSConnection("app.coqui.ai")
        url = self.MODEL_ENDPOINTS[self.model]["list_speakers"]
        conn.request("GET", f"{url}?page=1&per_page=100", headers=self.headers)
        res = conn.getresponse()
        data = res.read()
        return [Speaker(s) for s in json.loads(data)["result"]]

    def list_voices(self):
        """List custom voices created by the user."""
        conn = http.client.HTTPSConnection("app.coqui.ai")
        url = self.MODEL_ENDPOINTS[self.model]["list_voices"]
        conn.request("GET", f"{url}?page=1&per_page=100", headers=self.headers)
        res = conn.getresponse()
        data = res.read()
        return [Speaker(s, True) for s in json.loads(data)["result"]]

    def list_speakers_as_tts_models(self):
        """List speakers in ModelManager format."""
        models = []
        for speaker in self.speakers:
            model = f"coqui_studio/multilingual/{speaker.name}/{self.model}"
            models.append(model)
        return models

    def name_to_speaker(self, name):
        for speaker in self.speakers:
            if speaker.name == name:
                return speaker
        raise ValueError(f"Speaker {name} not found in {self.speakers}")

    def id_to_speaker(self, speaker_id):
        for speaker in self.speakers:
            if speaker.id == speaker_id:
                return speaker
        raise ValueError(f"Speaker {speaker_id} not found.")

    @staticmethod
    def url_to_np(url):
        tmp_file, _ = urllib.request.urlretrieve(url)
        rate, data = wavfile.read(tmp_file)
        return data, rate

    @staticmethod
    def _create_payload(model, text, speaker, speed, emotion, language):
        payload = {}
        # if speaker.is_voice:
        payload["voice_id"] = speaker.id
        # else:
        payload["speaker_id"] = speaker.id

        if model == "V1":
            payload.update(
                {
                    "emotion": emotion,
                    "name": speaker.name,
                    "text": text,
                    "speed": speed,
                }
            )
        elif model == "XTTS":
            payload.update(
                {
                    "name": speaker.name,
                    "text": text,
                    "speed": speed,
                    "language": language,
                }
            )
        else:
            raise ValueError(f"❗ Unknown model {model}")
        return payload

    def _check_tts_args(self, text, speaker_name, speaker_id, emotion, speed, language):
        assert text is not None, "❗ text is required for V1 model."
        assert speaker_name is not None, "❗ speaker_name is required for V1 model."
        if self.model == "V1":
            if emotion is None:
                emotion = "Neutral"
            assert language is None, "❗ language is not supported for V1 model."
        elif self.model == "XTTS":
            assert emotion is None, f"❗ Emotions are not supported for XTTS model. Use V1 model."
            assert language is not None, "❗ Language is required for XTTS model."
            assert (
                language in self.SUPPORTED_LANGUAGES
            ), f"❗ Language {language} is not yet supported. Check https://docs.coqui.ai/reference/samples_xtts_create."
        return text, speaker_name, speaker_id, emotion, speed, language

    def tts(
        self,
        text: str,
        speaker_name: str = None,
        speaker_id=None,
        emotion=None,
        speed=1.0,
        language=None,  # pylint: disable=unused-argument
    ) -> Tuple[np.ndarray, int]:
        """Synthesize speech from text.

        Args:
            text (str): Text to synthesize.
            speaker_name (str): Name of the speaker. You can get the list of speakers with `list_speakers()` and
                voices (user generated speakers) with `list_voices()`.
            speaker_id (str): Speaker ID. If None, the speaker name is used.
            emotion (str): Emotion of the speaker. One of "Neutral", "Happy", "Sad", "Angry", "Dull". Emotions are only
                supported by `V1` model. Defaults to None.
            speed (float): Speed of the speech. 1.0 is normal speed.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model. See https://docs.coqui.ai/reference/samples_xtts_create for supported languages.
        """
        self._check_token()
        self.ping_api()

        if speaker_name is None and speaker_id is None:
            raise ValueError(" [!] Please provide either a `speaker_name` or a `speaker_id`.")
        if speaker_id is None:
            speaker = self.name_to_speaker(speaker_name)
        else:
            speaker = self.id_to_speaker(speaker_id)

        text, speaker_name, speaker_id, emotion, speed, language = self._check_tts_args(
            text, speaker_name, speaker_id, emotion, speed, language
        )

        conn = http.client.HTTPSConnection("app.coqui.ai")
        payload = self._create_payload(self.model, text, speaker, speed, emotion, language)
        url = self.MODEL_ENDPOINTS[self.model]["synthesize"]
        conn.request("POST", url, json.dumps(payload), self.headers)
        res = conn.getresponse()
        data = res.read()
        try:
            wav, sr = self.url_to_np(json.loads(data)["audio_url"])
        except KeyError as e:
            raise ValueError(f" [!] 🐸 API returned error: {data}") from e
        return wav, sr

    def tts_to_file(
        self,
        text: str,
        speaker_name: str,
        speaker_id=None,
        emotion=None,
        speed=1.0,
        pipe_out=None,
        language=None,
        file_path: str = None,
    ) -> str:
        """Synthesize speech from text and save it to a file.

        Args:
            text (str): Text to synthesize.
            speaker_name (str): Name of the speaker. You can get the list of speakers with `list_speakers()` and
                voices (user generated speakers) with `list_voices()`.
            speaker_id (str): Speaker ID. If None, the speaker name is used.
            emotion (str): Emotion of the speaker. One of "Neutral", "Happy", "Sad", "Angry", "Dull".
            speed (float): Speed of the speech. 1.0 is normal speed.
            pipe_out (BytesIO, optional): Flag to stdout the generated TTS wav file for shell pipe.
            language (str): Language of the text. If None, the default language of the speaker is used. Language is only
                supported by `XTTS` model. Currently supports en, de, es, fr, it, pt, pl. Defaults to "en".
            file_path (str): Path to save the file. If None, a temporary file is created.
        """
        if file_path is None:
            file_path = tempfile.mktemp(".wav")
        wav, sr = self.tts(text, speaker_name, speaker_id, emotion, speed, language)
        save_wav(wav=wav, path=file_path, sample_rate=sr, pipe_out=pipe_out)
        return file_path


if __name__ == "__main__":
    import time

    api = CS_API()
    print(api.speakers)
    print(api.list_speakers_as_tts_models())

    ts = time.time()
    wav, sr = api.tts(
        "It took me quite a long time to develop a voice.", language="en", speaker_name=api.speakers[0].name
    )
    print(f" [i] XTTS took {time.time() - ts:.2f}s")

    filepath = api.tts_to_file(
        text="Hello world!", speaker_name=api.speakers[0].name, language="en", file_path="output.wav"
    )
