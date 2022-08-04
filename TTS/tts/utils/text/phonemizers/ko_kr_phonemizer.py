from typing import Dict
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.korean.phonemizer import korean_text_to_phonemes

_DEF_KO_PUNCS = "、.,[]()?!〽~『』「」【】"


class KO_KR_Phonemizer(BasePhonemizer):
    """🐸TTS ko_kr_phonemizer using functions in `TTS.tts.utils.text.korean.phonemizer`

    TODO: Add Korean to character (ᄀᄁᄂᄃᄄᄅᄆᄇᄈᄉᄊᄋᄌᄍᄎᄏᄐᄑ하ᅢᅣᅤᅥᅦᅧᅨᅩᅪᅫᅬᅭᅮᅯᅰᅱᅲᅳᅴᅵᆨᆩᆪᆫᆬᆭᆮᆯᆰᆱᆲᆳᆴᆵᆶᆷᆸᆹᆺᆻᆼᆽᆾᆿᇀᇁᇂ)

    Example:

        >>> from TTS.tts.utils.text.phonemizers import KO_KR_Phonemizer
        >>> phonemizer = KO_KR_Phonemizer()
        >>> phonemizer.phonemize("이 문장은 음성합성 테스트를 위한 문장입니다.", separator="|")
        'ᄋ|ᅵ| |ᄆ|ᅮ|ᆫ|ᄌ|ᅡ|ᆼ|ᄋ|ᅳ|ᆫ| |ᄋ|ᅳ|ᆷ|ᄉ|ᅥ|ᆼ|ᄒ|ᅡ|ᆸ|ᄉ|ᅥ|ᆼ| |ᄐ|ᅦ|ᄉ|ᅳ|ᄐ|ᅳ|ᄅ|ᅳ|ᆯ| |ᄋ|ᅱ|ᄒ|ᅡ|ᆫ| |ᄆ|ᅮ|ᆫ|ᄌ|ᅡ|ᆼ|ᄋ|ᅵ|ᆸ|ᄂ|ᅵ|ᄃ|ᅡ|.'

    """

    language = "ko-kr"

    def __init__(self, punctuations=_DEF_KO_PUNCS, keep_puncs=True, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "ko_kr_phonemizer"

    def _phonemize(self, text: str, separator: str = "") -> str:
        pass

    def phonemize(self, text: str, separator: str = "|") -> str:
        return korean_text_to_phonemes(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"ko-kr": "hangeul(korean)"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True



#if __name__ == "__main__":
#    test_text = "이 문장은 음성합성 테스트를 위한 문장입니다."
#    e = KO_KR_Phonemizer()
#    print(e.supported_languages())
#    print(e.version())
#    print(e.language)
#    print(e.name())
#    print(e.is_available())
#    print(e.phonemize(test_text))
