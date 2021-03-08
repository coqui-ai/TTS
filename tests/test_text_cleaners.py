#!/usr/bin/env python3

from TTS.tts.utils.text.cleaners import english_cleaners, phoneme_cleaners


def test_time() -> None:
    assert english_cleaners("It's 11:00") == "it's eleven a m"
    assert english_cleaners("It's 9:01") == "it's nine oh one a m"
    assert english_cleaners("It's 16:00") == "it's four p m"
    assert english_cleaners("It's 00:00 am") == "it's twelve a m"


def test_currency() -> None:
    assert phoneme_cleaners("It's $10.50") == "It's ten dollars fifty cents"
    assert phoneme_cleaners("£1.1") == "one pound sterling one penny"
    assert phoneme_cleaners("¥1") == "one yen"


def test_expand_numbers() -> None:
    assert phoneme_cleaners("-1") == 'minus one'
    assert phoneme_cleaners("1") == 'one'
