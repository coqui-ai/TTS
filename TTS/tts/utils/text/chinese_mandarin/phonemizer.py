from typing import List

import pypinyin

from .pinyinToPhonemes import PINYIN_DICT


def _chinese_character_to_pinyin(text: str) -> List[str]:
    pinyins = pypinyin.pinyin(
        text,
        style=pypinyin.Style.TONE3,
        heteronym=False,
        neutral_tone_with_five=True,
        # Add temp non-pinyin flag
        errors=lambda x: "￥" + x,
    )
    pinyins_flat_list = [item for sublist in pinyins for item in sublist]
    return pinyins_flat_list


def _chinese_pinyin_to_phoneme(pinyin: str) -> str:
    segment = pinyin[:-1]
    tone = pinyin[-1]
    phoneme = PINYIN_DICT.get(segment, [""])[0]
    return phoneme + tone


def is_pinyin(token: str):
    return not token.startswith("￥")


def chinese_text_to_phonemes(text: str, seperator: str = "|") -> str:
    pinyined_text: List[str] = _chinese_character_to_pinyin(text)

    results: List[str] = []

    for token in pinyined_text:
        if is_pinyin(token):
            pinyin_phonemes = _chinese_pinyin_to_phoneme(token)
            results += list(pinyin_phonemes)
        else:  # is ponctuation or other
            results += list(token[1:])  # remove the temp flag

        results.append(" ")

    return seperator.join(results).strip()
