from jamo import hangul_to_jamo

from TTS.tts.utils.text.korean.korean import normalize


def korean_text_to_phonemes(text):
    '''

        The input and output values look the same, but they are different in Unicode.

        example :

            input = '하늘' (Unicode : \ud558\ub298), (하 + 늘)
            output = '하늘' (Unicode :\u1112\u1161\u1102\u1173\u11af), (ᄒ + ᅡ + ᄂ + ᅳ + ᆯ)

    '''
    text = normalize(text)
    text = list(hangul_to_jamo(text))  # '하늘' --> ['ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆯ']
    return ''.join(text)


if __name__ == "__main__":
    print(korean_text_to_phonemes("테스트용 문장입니다."))
