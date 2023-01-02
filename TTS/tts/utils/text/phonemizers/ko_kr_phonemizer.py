from typing import Dict

from TTS.tts.utils.text.korean.phonemizer import korean_text_to_phonemes
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer

_DEF_KO_PUNCS = "、.,[]()?!〽~『』「」【】"


class KO_KR_Phonemizer(BasePhonemizer):
    """🐸TTS ko_kr_phonemizer using functions in `TTS.tts.utils.text.korean.phonemizer`

    TODO: Add Korean to character (ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ)

    Example:

        >>> from TTS.tts.utils.text.phonemizers import KO_KR_Phonemizer
        >>> phonemizer = KO_KR_Phonemizer()
        >>> phonemizer.phonemize("이 문장은 음성합성 테스트를 위한 문장입니다.", separator="|")
        'ᄋ|ᅵ| |ᄆ|ᅮ|ᆫ|ᄌ|ᅡ|ᆼ|ᄋ|ᅳ| |ᄂ|ᅳ|ᆷ|ᄉ|ᅥ|ᆼ|ᄒ|ᅡ|ᆸ|ᄊ|ᅥ|ᆼ| |ᄐ|ᅦ|ᄉ|ᅳ|ᄐ|ᅳ|ᄅ|ᅳ| |ᄅ|ᅱ|ᄒ|ᅡ|ᆫ| |ᄆ|ᅮ|ᆫ|ᄌ|ᅡ|ᆼ|ᄋ|ᅵ|ᆷ|ᄂ|ᅵ|ᄃ|ᅡ|.'

        >>> from TTS.tts.utils.text.phonemizers import KO_KR_Phonemizer
        >>> phonemizer = KO_KR_Phonemizer()
        >>> phonemizer.phonemize("이 문장은 음성합성 테스트를 위한 문장입니다.", separator="|", character='english')
        'I| |M|u|n|J|a|n|g|E|u| |N|e|u|m|S|e|o|n|g|H|a|b|S|s|e|o|n|g| |T|e|S|e|u|T|e|u|L|e|u| |L|w|i|H|a|n| |M|u|n|J|a|n|g|I|m|N|i|D|a|.'

    """

    language = "ko-kr"

    def __init__(self, punctuations=_DEF_KO_PUNCS, keep_puncs=True, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "ko_kr_phonemizer"

    def _phonemize(self, text: str, separator: str = "", character: str = "hangeul") -> str:
        ph = korean_text_to_phonemes(text, character=character)
        if separator is not None or separator != "":
            return separator.join(ph)
        return ph

    def phonemize(self, text: str, separator: str = "", character: str = "hangeul", language=None) -> str:
        return self._phonemize(text, separator, character)

    @staticmethod
    def supported_languages() -> Dict:
        return {"ko-kr": "hangeul(korean)"}

    def version(self) -> str:
        return "0.0.2"

    def is_available(self) -> bool:
        return True


if __name__ == "__main__":
    texts = "이 문장은 음성합성 테스트를 위한 문장입니다."
    e = KO_KR_Phonemizer()
    print(e.supported_languages())
    print(e.version())
    print(e.language)
    print(e.name())
    print(e.is_available())
    print(e.phonemize(texts))
