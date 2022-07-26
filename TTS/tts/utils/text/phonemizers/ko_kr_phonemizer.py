from typing import Dict
from TTS.tts.utils.text.phonemizers.base import BasePhonemizer
from TTS.tts.utils.text.korean.phonemizer import korean_text_to_phonemes
_DEF_JA_PUNCS = "、.,[]()?!〽~『』「」【】"


class KO_KR_Phonemizer(BasePhonemizer):
    '''
    2가지 옵션 존재
    만약 한글을 사용한다면 jamo를 사용해서 음소화작업을 거침
    만약 한글을 사용안한다면 transliteration작업을 통해 한글을 영어로 변환해서 출력
    '''

    language = "ko-kr"

    def __init__(self, punctuations=_DEF_JA_PUNCS, keep_puncs=True, **kwargs):  # pylint: disable=unused-argument
        super().__init__(self.language, punctuations=punctuations, keep_puncs=keep_puncs)

    @staticmethod
    def name():
        return "ko_kr_phonemizer"

    def _phonemize(self, text: str, separator: str = "|") -> str:
        pass

    def phonemize(self, text: str, separator: str ="|") -> str:
        return korean_text_to_phonemes(text, separator)

    @staticmethod
    def supported_languages() -> Dict:
        return {"ko-kr": "hangeul(korean)"}

    def version(self) -> str:
        return "0.0.1"

    def is_available(self) -> bool:
        return True


if __name__ == "__main__":
     text = "안녕하세요, 이 문장은 음성합성 테스트를 위한 문장입니다."
     e = KO_KR_Phonemizer()
     print(e.supported_languages())
     print(e.version())
     print(e.language)
     print(e.name())
     print(e.is_available())
     print(e.phonemize(text))