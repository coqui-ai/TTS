# coding: utf-8
# Code based on
# -*- coding: utf-8 -*-
# Copyright 2020 TensorFlowTTS Team, Jaehyoung Kim(@crux153) and Taehoon Kim(@carpedm20)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Code based on https://github.com/carpedm20/multi-speaker-tacotron-tensorflow
import re
import os
import ast
import json
from jamo import hangul_to_jamo, h2j, j2h
from jamo.jamo import _jamo_char_to_hcj

from .ko_dictionary import english_dictionary, etc_dictionary

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


def is_lead(char):
    return char in JAMO_LEADS


def is_vowel(char):
    return char in JAMO_VOWELS


def is_tail(char):
    return char in JAMO_TAILS


def get_mode(char):
    if is_lead(char):
        return 0
    elif is_vowel(char):
        return 1
    elif is_tail(char):
        return 2
    else:
        return -1


def _get_text_from_candidates(candidates):
    if len(candidates) == 0:
        return ""
    elif len(candidates) == 1:
        return _jamo_char_to_hcj(candidates[0])
    else:
        return j2h(**dict(zip(["lead", "vowel", "tail"], candidates)))


def jamo_to_korean(text):
    text = h2j(text)

    idx = 0
    new_text = ""
    candidates = []

    while True:
        if idx >= len(text):
            new_text += _get_text_from_candidates(candidates)
            break

        char = text[idx]
        mode = get_mode(char)

        if mode == 0:
            new_text += _get_text_from_candidates(candidates)
            candidates = [char]
        elif mode == -1:
            new_text += _get_text_from_candidates(candidates)
            new_text += char
            candidates = []
        else:
            candidates.append(char)

        idx += 1
    return new_text


num_to_kor = {
    '0': '영',
    '1': '일',
    '2': '이',
    '3': '삼',
    '4': '사',
    '5': '오',
    '6': '육',
    '7': '칠',
    '8': '팔',
    '9': '구',
}

unit_to_kor1 = {
    '%': '퍼센트',
    'cm': '센치미터',
    'mm': '밀리미터',
    'km': '킬로미터',
    'kg': '킬로그람',
}
unit_to_kor2 = {
    'm': '미터',
}

upper_to_kor = {
    'A': '에이',
    'B': '비',
    'C': '씨',
    'D': '디',
    'E': '이',
    'F': '에프',
    'G': '지',
    'H': '에이치',
    'I': '아이',
    'J': '제이',
    'K': '케이',
    'L': '엘',
    'M': '엠',
    'N': '엔',
    'O': '오',
    'P': '피',
    'Q': '큐',
    'R': '알',
    'S': '에스',
    'T': '티',
    'U': '유',
    'V': '브이',
    'W': '더블유',
    'X': '엑스',
    'Y': '와이',
    'Z': '지',
}


def compare_sentence_with_jamo(text1, text2):
    return h2j(text1) != h2j(text2)


def tokenize(text, as_id=False):
    # jamo package에 있는 hangul_to_jamo를 이용하여 한글 string을 초성/중성/종성으로 나눈다.
    text = normalize(text)
    tokens = list(hangul_to_jamo(text))  # '존경하는'  --> ['ᄌ', 'ᅩ', 'ᆫ', 'ᄀ', 'ᅧ', 'ᆼ', 'ᄒ', 'ᅡ', 'ᄂ', 'ᅳ', 'ᆫ', '~']

    if as_id:
        return [char_to_id[token] for token in tokens] + [char_to_id[EOS]]
    else:
        return [token for token in tokens] + [EOS]


def tokenizer_fn(iterator):
    return (token for x in iterator for token in tokenize(x, as_id=False))


def normalize(text):
    text = text.strip()

    text = re.sub('\(\d+일\)', '', text)
    text = re.sub('\([⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]+\)', '', text)

    text = normalize_with_dictionary(text, etc_dictionary)
    text = normalize_english(text)
    text = re.sub('[a-zA-Z]+', normalize_upper, text)

    text = normalize_quote(text)
    text = normalize_number(text)

    return text


def normalize_with_dictionary(text, dic):
    if any(key in text for key in dic.keys()):
        pattern = re.compile('|'.join(re.escape(key) for key in dic.keys()))
        return pattern.sub(lambda x: dic[x.group()], text)
    else:
        return text


def normalize_english(text):
    def fn(m):
        word = m.group()
        if word in english_dictionary:
            return english_dictionary.get(word)
        else:
            return word

    text = re.sub("([A-Za-z]+)", fn, text)
    return text


def normalize_upper(text):
    text = text.group(0)

    if all([char.isupper() for char in text]):
        return "".join(upper_to_kor[char] for char in text)
    else:
        return text


def normalize_quote(text):
    def fn(found_text):
        from nltk import sent_tokenize  # NLTK doesn't along with multiprocessing

        found_text = found_text.group()
        unquoted_text = found_text[1:-1]

        sentences = sent_tokenize(unquoted_text)
        return " ".join(["'{}'".format(sent) for sent in sentences])

    return re.sub(quote_checker, fn, text)


number_checker = "([+-]?\d[\d,]*)[\.]?\d*"
count_checker = "(시|명|가지|살|마리|포기|송이|수|톨|통|점|개|벌|척|채|다발|그루|자루|줄|켤레|그릇|잔|마디|상자|사람|곡|병|판)"


def normalize_number(text):
    text = normalize_with_dictionary(text, unit_to_kor1)
    text = normalize_with_dictionary(text, unit_to_kor2)
    text = re.sub(number_checker + count_checker,
                  lambda x: number_to_korean(x, True), text)
    text = re.sub(number_checker,
                  lambda x: number_to_korean(x, False), text)
    return text


num_to_kor1 = [""] + list("일이삼사오육칠팔구")
num_to_kor2 = [""] + list("만억조경해")
num_to_kor3 = [""] + list("십백천")

# count_to_kor1 = [""] + ["하나","둘","셋","넷","다섯","여섯","일곱","여덟","아홉"]
count_to_kor1 = [""] + ["한", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉"]

count_tenth_dict = {
    "십": "열",
    "두십": "스물",
    "세십": "서른",
    "네십": "마흔",
    "다섯십": "쉰",
    "여섯십": "예순",
    "일곱십": "일흔",
    "여덟십": "여든",
    "아홉십": "아흔",
}


def number_to_korean(num_str, is_count=False):
    if is_count:
        num_str, unit_str = num_str.group(1), num_str.group(2)
    else:
        num_str, unit_str = num_str.group(), ""

    num_str = num_str.replace(',', '')
    num = ast.literal_eval(num_str)

    if num == 0:
        return "영"

    check_float = num_str.split('.')
    if len(check_float) == 2:
        digit_str, float_str = check_float
    elif len(check_float) >= 3:
        raise Exception(" [!] Wrong number format")
    else:
        digit_str, float_str = check_float[0], None

    if is_count and float_str is not None:
        raise Exception(" [!] `is_count` and float number does not fit each other")

    digit = int(digit_str)

    if digit_str.startswith("-"):
        digit, digit_str = abs(digit), str(abs(digit))

    kor = ""
    size = len(str(digit))
    tmp = []

    for i, v in enumerate(digit_str, start=1):
        v = int(v)

        if v != 0:
            if is_count:
                tmp += count_to_kor1[v]
            else:
                tmp += num_to_kor1[v]

            tmp += num_to_kor3[(size - i) % 4]

        if (size - i) % 4 == 0 and len(tmp) != 0:
            kor += "".join(tmp)
            tmp = []
            kor += num_to_kor2[int((size - i) / 4)]

    if is_count:
        if kor.startswith("한") and len(kor) > 1:
            kor = kor[1:]

        if any(word in kor for word in count_tenth_dict):
            kor = re.sub(
                '|'.join(count_tenth_dict.keys()),
                lambda x: count_tenth_dict[x.group()], kor)

    if not is_count and kor.startswith("일") and len(kor) > 1:
        kor = kor[1:]

    if float_str is not None:
        kor += "쩜 "
        kor += re.sub('\d', lambda x: num_to_kor[x.group()], float_str)

    if num_str.startswith("+"):
        kor = "플러스 " + kor
    elif num_str.startswith("-"):
        kor = "마이너스 " + kor

    return kor + unit_str


if __name__ == "__main__":
    def test_normalize(text):
        print(text)
        print(normalize(text))
        print("=" * 30)


    test_normalize("JTBC는 JTBCs를 DY는 A가 Absolute")
    test_normalize("오늘(13일) 3,600마리 강아지가")
    test_normalize("60.3%")
    test_normalize('"저돌"(猪突) 입니다.')
    test_normalize('비대위원장이 지난 1월 이런 말을 했습니다. “난 그냥 산돼지처럼 돌파하는 스타일이다”')
    test_normalize("지금은 -12.35%였고 종류는 5가지와 19가지, 그리고 55가지였다")
    test_normalize("JTBC는 TH와 K 양이 2017년 9월 12일 오후 12시에 24살이 된다")
    print(list(hangul_to_jamo(list(hangul_to_jamo('비대위원장이 지난 1월 이런 말을 했습니다? “난 그냥 산돼지처럼 돌파하는 스타일이다”')))))
