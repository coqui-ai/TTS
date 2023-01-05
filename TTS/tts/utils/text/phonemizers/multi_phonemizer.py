from typing import Dict, List

from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name


class MultiPhonemizer:
    """🐸TTS multi-phonemizer that operates phonemizers for multiple langugages

    Args:
        custom_lang_to_phonemizer (Dict):
            Custom phonemizer mapping if you want to change the defaults. In the format of
            `{"lang_code", "phonemizer_name"}`. When it is None, `DEF_LANG_TO_PHONEMIZER` is used. Defaults to `{}`.

    TODO: find a way to pass custom kwargs to the phonemizers
    """

    lang_to_phonemizer = {}

    def __init__(self, lang_to_phonemizer_name: Dict = {}) -> None:  # pylint: disable=dangerous-default-value
        for k, v in lang_to_phonemizer_name.items():
            if v == "" and k in DEF_LANG_TO_PHONEMIZER.keys():
                lang_to_phonemizer_name[k] = DEF_LANG_TO_PHONEMIZER[k]
            elif v == "":
                raise ValueError(f"Phonemizer wasn't set for language {k} and doesn't have a default.")
        self.lang_to_phonemizer_name = lang_to_phonemizer_name
        self.lang_to_phonemizer = self.init_phonemizers(self.lang_to_phonemizer_name)

    @staticmethod
    def init_phonemizers(lang_to_phonemizer_name: Dict) -> Dict:
        lang_to_phonemizer = {}
        for k, v in lang_to_phonemizer_name.items():
            lang_to_phonemizer[k] = get_phonemizer_by_name(v, language=k)
        return lang_to_phonemizer

    @staticmethod
    def name():
        return "multi-phonemizer"

    def phonemize(self, text, separator="|", language=""):
        if language == "":
            raise ValueError("Language must be set for multi-phonemizer to phonemize.")
        return self.lang_to_phonemizer[language].phonemize(text, separator)

    def supported_languages(self) -> List:
        return list(self.lang_to_phonemizer.keys())

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.supported_languages()}")
        print(f"{indent}| > phoneme backend: {self.name()}")


# if __name__ == "__main__":
#     texts = {
#         "tr": "Merhaba, bu Türkçe bit örnek!",
#         "en-us": "Hello, this is English example!",
#         "de": "Hallo, das ist ein Deutches Beipiel!",
#         "zh-cn": "这是中国的例子",
#     }
#     phonemes = {}
#     ph = MultiPhonemizer({"tr": "espeak", "en-us": "", "de": "gruut", "zh-cn": ""})
#     for lang, text in texts.items():
#         phoneme = ph.phonemize(text, lang)
#         phonemes[lang] = phoneme
#     print(phonemes)
