"""Tests for text to phoneme converstion"""
import unittest

import gruut
from gruut_ipa import IPA, Phonemes

from TTS.tts.utils.text import clean_gruut_phonemes, phoneme_to_sequence
from TTS.tts.utils.text import phonemes as all_phonemes
from TTS.tts.utils.text import sequence_to_phoneme

# -----------------------------------------------------------------------------

EXAMPLE_TEXT = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"

# Raw phonemes from run of gruut with example text (en-us).
# This includes IPA ties, etc.
EXAMPLE_PHONEMES = [
    ["ɹ", "ˈi", "s", "ə", "n", "t"],
    ["ɹ", "i", "s", "ˈɚ", "t͡ʃ"],
    ["ˈæ", "t"],
    ["h", "ˈɑ", "ɹ", "v", "ɚ", "d"],
    ["h", "ˈæ", "z"],
    ["ʃ", "ˈoʊ", "n"],
    ["m", "ˈɛ", "d", "ɪ", "t", "ˌeɪ", "t", "ɪ", "ŋ"],
    ["f", "ɚ"],
    ["ˈæ", "z"],
    ["l", "ˈɪ", "t", "ə", "l"],
    ["ˈæ", "z"],
    ["ˈeɪ", "t"],
    ["w", "ˈi", "k", "s"],
    ["k", "ə", "n"],
    ["ˈæ", "k", "t͡ʃ", "ə", "l", "i"],
    ["ɪ", "ŋ", "k", "ɹ", "ˈi", "s"],
    [","],
    ["ð", "ə"],
    ["ɡ", "ɹ", "ˈeɪ"],
    ["m", "ˈæ", "t", "ɚ"],
    ["ˈɪ", "n"],
    ["ð", "ə"],
    ["p", "ˈɑ", "ɹ", "t", "s"],
    ["ə", "v"],
    ["ð", "ə"],
    ["b", "ɹ", "ˈeɪ", "n"],
    ["ɹ", "i", "s", "p", "ˈɑ", "n", "s", "ɪ", "b", "ə", "l"],
    ["f", "ɚ"],
    ["ɪ", "m", "ˈoʊ", "ʃ", "ə", "n", "ə", "l"],
    ["ɹ", "ˌɛ", "ɡ", "j", "ə", "l", "ˈeɪ", "ʃ", "ə", "n"],
    ["ˈæ", "n", "d"],
    ["l", "ˈɚ", "n", "ɪ", "ŋ"],
    ["!"],
]

# -----------------------------------------------------------------------------


class TextProcessingTextCase(unittest.TestCase):
    """Tests for text to phoneme conversion"""

    def test_all_phonemes_in_tts(self):
        """Ensure that all phonemes from gruut are present in TTS phonemes"""
        tts_phonemes = set(all_phonemes)

        # Check stress characters
        for suprasegmental in [IPA.STRESS_PRIMARY, IPA.STRESS_SECONDARY]:
            self.assertIn(suprasegmental, tts_phonemes)

        # Check that gruut's phonemes are a subset of TTS phonemes
        for lang in gruut.get_supported_languages():
            for phoneme in Phonemes.from_language(lang):
                for codepoint in clean_gruut_phonemes(phoneme.text):

                    self.assertIn(codepoint, tts_phonemes)

    def test_phoneme_to_sequence(self):
        """Verify example (text -> sequence -> phoneme string) pipeline"""
        lang = "en-us"
        expected_phoneme_str = " ".join(
            "".join(clean_gruut_phonemes(word_phonemes)) for word_phonemes in EXAMPLE_PHONEMES
        )

        # Ensure that TTS produces same phoneme string
        text_cleaner = ["phoneme_cleaners"]
        actual_sequence = phoneme_to_sequence(EXAMPLE_TEXT, text_cleaner, lang)
        actual_phoneme_str = sequence_to_phoneme(actual_sequence)

        self.assertEqual(actual_phoneme_str, expected_phoneme_str)

    def test_phoneme_to_sequence_with_blank_token(self):
        """Verify example (text -> sequence -> phoneme string) pipeline with blank token"""
        lang = "en-us"
        text_cleaner = ["phoneme_cleaners"]

        # Create with/without blank sequences
        sequence_without_blank = phoneme_to_sequence(EXAMPLE_TEXT, text_cleaner, lang, add_blank=False)
        sequence_with_blank = phoneme_to_sequence(EXAMPLE_TEXT, text_cleaner, lang, add_blank=True)

        # With blank sequence should be bigger
        self.assertGreater(len(sequence_with_blank), len(sequence_without_blank))

        # But phoneme strings should still be identical
        phoneme_str_without_blank = sequence_to_phoneme(sequence_without_blank, add_blank=False)
        phoneme_str_with_blank = sequence_to_phoneme(sequence_with_blank, add_blank=True)

        self.assertEqual(phoneme_str_with_blank, phoneme_str_without_blank)

    def test_messy_text(self):
        """Verify text with extra punctuation/whitespace/etc. makes it through the pipeline"""
        text = '"Be" a! voice, [NOT]? (an eCHo.   '
        lang = "en-us"
        expected_phonemes = [
            ["b", "ˈi"],
            ["ə"],
            ["!"],
            ["v", "ˈɔɪ", "s"],
            [","],
            ["n", "ˈɑ", "t"],
            ["?"],
            ["ə", "n"],
            ["ˈɛ", "k", "oʊ"],
            ["."],
        ]
        expected_phoneme_str = " ".join(
            "".join(clean_gruut_phonemes(word_phonemes)) for word_phonemes in expected_phonemes
        )

        # Ensure that TTS produces same phoneme string
        text_cleaner = ["phoneme_cleaners"]
        actual_sequence = phoneme_to_sequence(text, text_cleaner, lang)
        actual_phoneme_str = sequence_to_phoneme(actual_sequence)

        self.assertEqual(actual_phoneme_str, expected_phoneme_str)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
