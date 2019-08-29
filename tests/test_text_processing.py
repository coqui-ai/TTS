import unittest
import torch as T

from TTS.utils.text import *

def test_phoneme_to_sequence():
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    text_cleaner = ["phoneme_cleaners"]
    lang = "en-us"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    gt = "ɹiːsənt ɹɪsɜːtʃ æt hɑːɹvɚd hɐz ʃoʊn mɛdᵻteɪɾɪŋ fɔːɹ æz lɪɾəl æz eɪt wiːks kæn æktʃuːəli ɪnkɹiːs, ðə ɡɹeɪ mæɾɚɹ ɪnðə pɑːɹts ʌvðə bɹeɪn ɹɪspɑːnsəbəl fɔːɹ ɪmoʊʃənəl ɹɛɡjuːleɪʃən ænd lɜːnɪŋ!"
    assert text_hat == gt

    # multiple punctuations
    text = "Be a voice, not an! echo?"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ?"
    print(text_hat)
    print(len(sequence))
    assert text_hat == gt

    # not ending with punctuation
    text = "Be a voice, not an! echo"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == gt

    # original
    text = "Be a voice, not an echo!"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    gt = "biː ɐ vɔɪs, nɑːt ɐn ɛkoʊ!"
    print(text_hat)
    print(len(sequence))
    assert text_hat == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ."
    print(text_hat)
    print(len(sequence))
    assert text_hat == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang, True)
    text_hat = sequence_to_phoneme(sequence)
    gt = "^biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ.~"
    print(text_hat)
    print(len(sequence))
    assert text_hat == gt

    # padding char
    text = "_Be a _voice, not an! echo_"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == gt


def test_text2phone():
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    gt = "ɹ|iː|s|ə|n|t| |ɹ|ɪ|s|ɜː|tʃ| |æ|t| |h|ɑːɹ|v|ɚ|d| |h|ɐ|z| |ʃ|oʊ|n| |m|ɛ|d|ᵻ|t|eɪ|ɾ|ɪ|ŋ| |f|ɔː|ɹ| |æ|z| |l|ɪ|ɾ|əl| |æ|z| |eɪ|t| |w|iː|k|s| |k|æ|n| |æ|k|tʃ|uː|əl|i|| |ɪ|n|k|ɹ|iː|s|,| |ð|ə| |ɡ|ɹ|eɪ| |m|æ|ɾ|ɚ|ɹ| |ɪ|n|ð|ə| |p|ɑːɹ|t|s| |ʌ|v|ð|ə| |b|ɹ|eɪ|n| |ɹ|ɪ|s|p|ɑː|n|s|ə|b|əl| |f|ɔː|ɹ| |ɪ|m|oʊ|ʃ|ə|n|əl| |ɹ|ɛ|ɡ|j|uː|l|eɪ|ʃ|ə|n||| |æ|n|d| |l|ɜː|n|ɪ|ŋ|!"
    lang = "en-us"
    phonemes = text2phone(text, lang)
    assert gt == phonemes
