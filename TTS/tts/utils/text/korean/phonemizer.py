import re
import os
import ast
import json
from jamo import hangul_to_jamo, h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from TTS.tts.utils.text.korean.ko_dictionary import english_dictionary, etc_dictionary

PAD = '_'
EOS = '~'
PUNC = '!\'(),-.:;?'
SPACE = ' '

JAMO_LEADS = "".join([chr(_) for _ in range(0x1100, 0x1113)])
JAMO_VOWELS = "".join([chr(_) for _ in range(0x1161, 0x1176)])
JAMO_TAILS = "".join([chr(_) for _ in range(0x11A8, 0x11C3)])

VALID_CHARS = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + PUNC + SPACE
ALL_SYMBOLS = PAD + EOS + VALID_CHARS

char_to_id = {c: i for i, c in enumerate(ALL_SYMBOLS)}
id_to_char = {i: c for i, c in enumerate(ALL_SYMBOLS)}

quote_checker = """([`"'＂“‘])(.+?)([`"'＂”’])"""


def korean_text_to_phonemes(text, separator, as_id=False):
    # jamo package에 있는 hangul_to_jamo를 이용하여 한글 string을 초성/중성/종성으로 나눈다.
    tokens = list(hangul_to_jamo(text))  # '존경하는'  --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']
    if as_id:
        return [char_to_id[token] for token in tokens]
    else:
        return(separator.join(tokens))


if __name__ == "__main__":
    print(korean_text_to_phonemes("i have a dream. it's a big dream", "|"))
    print(korean_text_to_phonemes("나는 밥을 ㅁ거을거아", " "))
    tokens = list(hangul_to_jamo("안녕 나는 밥이야"))
    for i in tokens:
        print(i)