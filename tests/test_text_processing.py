import os
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# pylint: disable=unused-import
import unittest
from TTS.utils.text import *
from TTS.tests import get_tests_path
from TTS.utils.generic_utils import load_config

TESTS_PATH = get_tests_path()
conf = load_config(os.path.join(TESTS_PATH, 'test_config.json'))

def test_phoneme_to_sequence():
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    text_cleaner = ["phoneme_cleaners"]
    lang = "en-us"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "ɹiːsənt ɹɪsɜːtʃ æt hɑːɹvɚd hɐz ʃoʊn mɛdᵻteɪɾɪŋ fɔːɹ æz lɪɾəl æz eɪt wiːks kæn æktʃuːəli ɪnkɹiːs, ðə ɡɹeɪ mæɾɚɹ ɪnðə pɑːɹts ʌvðə bɹeɪn ɹɪspɑːnsəbəl fɔːɹ ɪmoʊʃənəl ɹɛɡjuːleɪʃən ænd lɜːnɪŋ!"
    assert text_hat == text_hat_with_params == gt 

    # multiple punctuations
    text = "Be a voice, not an! echo?"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ?"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # not ending with punctuation
    text = "Be a voice, not an! echo"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # original
    text = "Be a voice, not an echo!"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt ɐn ɛkoʊ!"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ."
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang, True)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "^biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ.~"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # padding char
    text = "_Be a _voice, not an! echo_"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    sequence_with_params = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

def test_text2phone():
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    gt = "ɹ|iː|s|ə|n|t| |ɹ|ɪ|s|ɜː|tʃ| |æ|t| |h|ɑːɹ|v|ɚ|d| |h|ɐ|z| |ʃ|oʊ|n| |m|ɛ|d|ᵻ|t|eɪ|ɾ|ɪ|ŋ| |f|ɔː|ɹ| |æ|z| |l|ɪ|ɾ|əl| |æ|z| |eɪ|t| |w|iː|k|s| |k|æ|n| |æ|k|tʃ|uː|əl|i| |ɪ|n|k|ɹ|iː|s|,| |ð|ə| |ɡ|ɹ|eɪ| |m|æ|ɾ|ɚ|ɹ| |ɪ|n|ð|ə| |p|ɑːɹ|t|s| |ʌ|v|ð|ə| |b|ɹ|eɪ|n| |ɹ|ɪ|s|p|ɑː|n|s|ə|b|əl| |f|ɔː|ɹ| |ɪ|m|oʊ|ʃ|ə|n|əl| |ɹ|ɛ|ɡ|j|uː|l|eɪ|ʃ|ə|n| |æ|n|d| |l|ɜː|n|ɪ|ŋ|!"
    lang = "en-us"
    ph = text2phone(text, lang)
    assert gt == ph, f"\n{phonemes} \n vs \n{gt}"