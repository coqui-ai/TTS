import os
# pylint: disable=unused-wildcard-import
# pylint: disable=wildcard-import
# pylint: disable=unused-import
from tests import get_tests_input_path
from TTS.tts.utils.text import *
from tests import get_tests_path
from TTS.utils.io import load_config

conf = load_config(os.path.join(get_tests_input_path(), 'test_config.json'))

def test_phoneme_to_sequence():

    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    text_cleaner = ["phoneme_cleaners"]
    lang = "en-us"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = 'ɹiːsənt ɹᵻsɜːtʃ æt hɑːɹvɚd hɐz ʃoʊn mɛdᵻteɪɾɪŋ fɔːɹ æz lɪɾəl æz eɪt wiːks kæn æktʃuːəli ɪŋkɹiːs, ðə ɡɹeɪ mæɾɚɹ ɪnðə pɑːɹts ʌvðə bɹeɪn ɹᵻspɑːnsᵻbəl fɔːɹ ɪmoʊʃənəl ɹɛɡjʊleɪʃən ænd lɜːnɪŋ!'
    assert text_hat == text_hat_with_params == gt

    # multiple punctuations
    text = "Be a voice, not an! echo?"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt æn! ɛkoʊ?"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # not ending with punctuation
    text = "Be a voice, not an! echo"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt æn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # original
    text = "Be a voice, not an echo!"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt ɐn ɛkoʊ!"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt æn! ɛkoʊ."
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang, True)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "^biː ɐ vɔɪs, nɑːt æn! ɛkoʊ.~"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # padding char
    text = "_Be a _voice, not an! echo_"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters)
    gt = "biː ɐ vɔɪs, nɑːt æn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

def test_phoneme_to_sequence_with_blank_token():

    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    text_cleaner = ["phoneme_cleaners"]
    lang = "en-us"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = "ɹiːsənt ɹᵻsɜːtʃ æt hɑːɹvɚd hɐz ʃoʊn mɛdᵻteɪɾɪŋ fɔːɹ æz lɪɾəl æz eɪt wiːks kæn æktʃuːəli ɪŋkɹiːs, ðə ɡɹeɪ mæɾɚɹ ɪnðə pɑːɹts ʌvðə bɹeɪn ɹᵻspɑːnsᵻbəl fɔːɹ ɪmoʊʃənəl ɹɛɡjʊleɪʃən ænd lɜːnɪŋ!"
    assert text_hat == text_hat_with_params == gt

    # multiple punctuations
    text = "Be a voice, not an! echo?"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = 'biː ɐ vɔɪs, nɑːt æn! ɛkoʊ?'
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # not ending with punctuation
    text = "Be a voice, not an! echo"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = 'biː ɐ vɔɪs, nɑːt æn! ɛkoʊ'
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # original
    text = "Be a voice, not an echo!"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = 'biː ɐ vɔɪs, nɑːt ɐn ɛkoʊ!'
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = 'biː ɐ vɔɪs, nɑːt æn! ɛkoʊ.'
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # extra space after the sentence
    text = "Be a voice, not an! echo.  "
    sequence = phoneme_to_sequence(text, text_cleaner, lang, True)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = "^biː ɐ vɔɪs, nɑːt æn! ɛkoʊ.~"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

    # padding char
    text = "_Be a _voice, not an! echo_"
    sequence = phoneme_to_sequence(text, text_cleaner, lang)
    text_hat = sequence_to_phoneme(sequence)
    _ = phoneme_to_sequence(text, text_cleaner, lang, tp=conf.characters, add_blank=True)
    text_hat_with_params = sequence_to_phoneme(sequence, tp=conf.characters, add_blank=True)
    gt = "biː ɐ vɔɪs, nɑːt æn! ɛkoʊ"
    print(text_hat)
    print(len(sequence))
    assert text_hat == text_hat_with_params == gt

def test_text2phone():
    text = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"
    gt = 'ɹ|iː|s|ə|n|t| |ɹ|ᵻ|s|ɜː|tʃ| |æ|t| |h|ɑːɹ|v|ɚ|d| |h|ɐ|z| |ʃ|oʊ|n| |m|ɛ|d|ᵻ|t|eɪ|ɾ|ɪ|ŋ| |f|ɔː|ɹ| |æ|z| |l|ɪ|ɾ|əl| |æ|z| |eɪ|t| |w|iː|k|s| |k|æ|n| |æ|k|tʃ|uː|əl|i| |ɪ|ŋ|k|ɹ|iː|s|,| |ð|ə| |ɡ|ɹ|eɪ| |m|æ|ɾ|ɚ|ɹ| |ɪ|n|ð|ə| |p|ɑːɹ|t|s| |ʌ|v|ð|ə| |b|ɹ|eɪ|n| |ɹ|ᵻ|s|p|ɑː|n|s|ᵻ|b|əl| |f|ɔː|ɹ| |ɪ|m|oʊ|ʃ|ə|n|əl| |ɹ|ɛ|ɡ|j|ʊ|l|eɪ|ʃ|ə|n| |æ|n|d| |l|ɜː|n|ɪ|ŋ|!'
    lang = "en-us"
    ph = text2phone(text, lang)
    assert gt == ph
