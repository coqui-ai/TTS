from typing import Dict, List

#from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.phonemizers.espeak_wrapper import ESpeak
from TTS.tts.utils.text.phonemizers.gruut_wrapper import Gruut
from TTS.tts.utils.text.phonemizers.ja_jp_phonemizer import JA_JP_Phonemizer
from TTS.tts.utils.text.phonemizers.zh_cn_phonemizer import ZH_CN_Phonemizer
from TTS.tts.utils.text.phonemizers.ko_kr_phonemizer import KO_KR_Phonemizer

def get_phonemizer_by_name(name: str, **kwargs) -> BasePhonemizer:
    """Initiate a phonemizer by name

    Args:
        name (str):
            Name of the phonemizer that should match `phonemizer.name()`.

        kwargs (dict):
            Extra keyword arguments that should be passed to the phonemizer.
    """
    if name == "espeak":
        return ESpeak(**kwargs)
    if name == "gruut":
        return Gruut(**kwargs)
    if name == "zh_cn_phonemizer":
        return ZH_CN_Phonemizer(**kwargs)
    if name == "ja_jp_phonemizer":
        return JA_JP_Phonemizer(**kwargs)
    if name == "ko_kr_phonemizer":
        return KO_KR_Phonemizer(**kwargs)

    raise ValueError(f"Phonemizer {name} not found")


class MultiPhonemizer():
    """🐸TTS multi-phonemizer that operates phonemizers for multiple langugages

    Args:
        custom_lang_to_phonemizer (Dict):
            Custom phonemizer mapping if you want to change the defaults. In the format of
            `{"lang_code", "phonemizer_name"}`. When it is None, `DEF_LANG_TO_PHONEMIZER` is used. Defaults to `{}`.

    TODO: find a way to pass custom kwargs to the phonemizers
    """

    lang_to_phonemizer_name = {'ja-jp' : 'ja_jp_phonemizer', 'ko-kr' : 'ko_kr_phonemizer'}
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
        return "multi_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        pass

    def phonemize(self, text, language, separator="|"):
        return self.lang_to_phonemizer[language].phonemize(text, separator)


    def supported_languages(self) -> List:
        return list(self.lang_to_phonemizer_name.keys())

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > phoneme language: {self.language}")
        print(f"{indent}| > phoneme backend: {self.name()}")

    @staticmethod
    def supported_languages() -> Dict:
        return {"ko-kr": "hangeul(korean)" , "ja-jp": "japanese"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


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
