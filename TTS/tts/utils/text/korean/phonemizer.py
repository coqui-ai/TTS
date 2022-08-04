from TTS.tts.utils.text.korean.korean import normalize
from jamo import hangul_to_jamo


def korean_text_to_phonemes(text, separator):
    # jamo package에 있는 hangul_to_jamo를 이용하여 한글 string을 초성/중성/종성으로 나눈다.
    text = normalize(text)
    tokens = list(hangul_to_jamo(text))  # '존경하는'  --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']
    return separator.join(tokens)


if __name__ == "__main__":
    print(korean_text_to_phonemes("테스트용 문장입니다.", " "))
