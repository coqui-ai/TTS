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

    lang_to_phonemizer_name = DEF_LANG_TO_PHONEMIZER
    language = "multi-lingual"

    def __init__(self, custom_lang_to_phonemizer: Dict = {}) -> None:  # pylint: disable=dangerous-default-value
        self.lang_to_phonemizer_name.update(custom_lang_to_phonemizer)
        self.lang_to_phonemizer = self.init_phonemizers(self.lang_to_phonemizer_name)

    @staticmethod
    def init_phonemizers(lang_to_phonemizer_name: Dict) -> Dict:
        lang_to_phonemizer = {}
        for k, v in lang_to_phonemizer_name.items():
            phonemizer = get_phonemizer_by_name(v, language=k)
            lang_to_phonemizer[k] = phonemizer
        return lang_to_phonemizer

    @staticmethod
    def name():
        return "multi-phonemizer"

    def phonemize(self, text, language, separator="|"):
        return self.lang_to_phonemizer[language].phonemize(text, separator)

    def supported_languages(self) -> List:
        return list(self.lang_to_phonemizer_name.keys())


# if __name__ == "__main__":
#     texts = {
#         "tr": "Merhaba, bu Türkçe bit örnek!",
#         "en-us": "Hello, this is English example!",
#         "de": "Hallo, das ist ein Deutches Beipiel!",
#         "zh-cn": "这是中国的例子",
#     }
#     phonemes = {}
#     ph = MultiPhonemizer()
#     for lang, text in texts.items():
#         phoneme = ph.phonemize(text, lang)
#         phonemes[lang] = phoneme
#     print(phonemes)
