#!/usr/bin/env python3

from TTS.tts.utils.text.cleaners import english_cleaners, italian_cleaners, phoneme_cleaners


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
    assert phoneme_cleaners("-1") == "minus one"
    assert phoneme_cleaners("1") == "one"


def test_italian_cleaners_numbers_and_abbreviations() -> None:
    text = "Alle 09:05 Sig. Bianchi ha pagato €1.234,50 per 50%"
    expected = "alle 9 e cinque signor bianchi ha pagato 1234 virgola 50 euro per 50 percento"
    assert italian_cleaners(text) == expected


def test_italian_cleaners_temperature_and_time() -> None:
    text = "Temperatura: 3.5\N{DEGREE SIGN}C alle 14.00"
    expected = "temperatura, 3 virgola 5 gradi celsius alle 14"
    assert italian_cleaners(text) == expected
