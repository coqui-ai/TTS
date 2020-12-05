#!/usr/bin/env python3

from TTS.tts.utils.text.cleaners import english_cleaners, phoneme_cleaners
from typing import Any


def assert_equal(actual: Any, expected: Any) -> None:
    assert actual == expected, f"\n{actual} \n vs \n{expected}"


def test_time() -> None:
    assert_equal(english_cleaners("It's 11:00"), "it's eleven a m")
    assert_equal(english_cleaners("It's 9:01"), "it's nine oh one a m")
    assert_equal(english_cleaners("It's 16:00"), "it's four p m")
    assert_equal(english_cleaners("It's 00:00 am"), "it's twelve a m")


def test_currency() -> None:
    assert_equal(phoneme_cleaners("It's $10.50"),
                 "It's ten dollars fifty cents")
    assert_equal(phoneme_cleaners("£1.1"),
                 "one pound sterling one penny")
    assert_equal(phoneme_cleaners("¥1"),
                 "one yen")
