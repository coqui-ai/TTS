"""Tests for text to phoneme converstion"""
import unittest

from TTS.tts.utils.text import phoneme_to_sequence, sequence_to_phoneme, text2phone

# -----------------------------------------------------------------------------

LANG = "en-us"

EXAMPLE_TEXT = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"

EXPECTED_PHONEMES = "ɹ|i|ː|s|ə|n|t| ɹ|ᵻ|s|ɜ|ː|t|ʃ| æ|ɾ| h|ɑ|ː|ɹ|v|ɚ|d| h|ɐ|z| ʃ|o|ʊ|n| m|ɛ|d|ᵻ|t|e|ɪ|ɾ|ɪ|ŋ| f|ɔ|ː|ɹ| æ|z| l|ɪ|ɾ|ə|l| æ|z| e|ɪ|t| w|i|ː|k|s| k|æ|ŋ| æ|k|t|ʃ|u|ː|ə|l|i| ɪ|ŋ|k|ɹ|i|ː|s|,| ð|ə| ɡ|ɹ|e|ɪ| m|æ|ɾ|ɚ| ɪ|n| ð|ə| p|ɑ|ː|ɹ|t|s| ʌ|v| ð|ə| b|ɹ|e|ɪ|n| ɹ|ᵻ|s|p|ɑ|ː|n|s|ᵻ|b|ə|l| f|ɔ|ː|ɹ| ɪ|m|o|ʊ|ʃ|ə|n|ə|l| ɹ|ɛ|ɡ|j|ʊ|l|e|ɪ|ʃ|ə|n| æ|n|d| l|ɜ|ː|n|ɪ|ŋ|!"

# -----------------------------------------------------------------------------


class TextProcessingTestCase(unittest.TestCase):
    """Tests for text to phoneme conversion"""

    def test_phoneme_to_sequence(self):
        """Verify en-us sentence phonemes without blank token"""
        self._test_phoneme_to_sequence(add_blank=False)

    def test_phoneme_to_sequence_with_blank_token(self):
        """Verify en-us sentence phonemes with blank token"""
        self._test_phoneme_to_sequence(add_blank=True)

    def _test_phoneme_to_sequence(self, add_blank):
        """Verify en-us sentence phonemes"""
        text_cleaner = ["phoneme_cleaners"]
        sequence = phoneme_to_sequence(EXAMPLE_TEXT, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = EXPECTED_PHONEMES.replace("|", "")
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # multiple punctuations
        text = "Be a voice, not an! echo?"
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ?"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # not ending with punctuation
        text = "Be a voice, not an! echo"
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # original
        text = "Be a voice, not an echo!"
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs, nɑːt ɐn ɛkoʊ!"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # extra space after the sentence
        text = "Be a voice, not an! echo.  "
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ."
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # extra space after the sentence
        text = "Be a voice, not an! echo.  "
        sequence = phoneme_to_sequence(
            text, text_cleaner, LANG, enable_eos_bos=True, add_blank=add_blank, use_espeak_phonemes=True
        )
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "^biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ.~"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

    def test_text2phone(self):
        """Verify phones directly (with |)"""
        ph = text2phone(EXAMPLE_TEXT, LANG, use_espeak_phonemes=True)
        self.assertEqual(ph, EXPECTED_PHONEMES)


# -----------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main()
