"""Tests for text to phoneme converstion"""
import unittest

from TTS.tts.utils.text import phoneme_to_sequence, sequence_to_phoneme, text2phone

# -----------------------------------------------------------------------------

LANG = "en-us"

EXAMPLE_TEXT = "Recent research at Harvard has shown meditating for as little as 8 weeks can actually increase, the grey matter in the parts of the brain responsible for emotional regulation and learning!"

EXPECTED_PHONEMES = "ɹ|iː|s|ə|n|t| ɹ|ᵻ|s|ɜː|tʃ| æ|t| h|ɑːɹ|v|ɚ|d| h|æ|z| ʃ|oʊ|n| m|ɛ|d|ᵻ|t|eɪ|ɾ|ɪ|ŋ| f|ɔːɹ| æ|z| l|ɪ|ɾ|əl| æ|z| eɪ|t| w|iː|k|s| k|æ|n| æ|k|tʃ|uː|əl|i| ɪ|ŋ|k|ɹ|iː|s| ,| ð|ə| ɡ|ɹ|eɪ| m|æ|ɾ|ɚ| ɪ|n| ð|ə| p|ɑːɹ|t|s| ʌ|v| ð|ə| b|ɹ|eɪ|n| ɹ|ᵻ|s|p|ɑː|n|s|ᵻ|b|əl| f|ɔːɹ| ɪ|m|oʊ|ʃ|ə|n|əl| ɹ|ɛ|ɡ|j|ʊ|l|eɪ|ʃ|ə|n| æ|n|d| l|ɜː|n|ɪ|ŋ| !"

# -----------------------------------------------------------------------------


class TextProcessingTextCase(unittest.TestCase):
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
        gt = "biː ɐ vɔɪs , nɑːt ɐn ! ɛkoʊ ?"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # not ending with punctuation
        text = "Be a voice, not an! echo"
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs , nɑːt ɐn ! ɛkoʊ"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # original
        text = "Be a voice, not an echo!"
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs , nɑːt ɐn ɛkoʊ !"
        print(text_hat)
        print(len(sequence))
        self.assertEqual(text_hat, text_hat_with_params)
        self.assertEqual(text_hat, gt)

        # extra space after the sentence
        text = "Be a voice, not an! echo.  "
        sequence = phoneme_to_sequence(text, text_cleaner, LANG, add_blank=add_blank, use_espeak_phonemes=True)
        text_hat = sequence_to_phoneme(sequence)
        text_hat_with_params = sequence_to_phoneme(sequence)
        gt = "biː ɐ vɔɪs , nɑːt ɐn ! ɛkoʊ ."
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
        gt = "^biː ɐ vɔɪs , nɑːt ɐn ! ɛkoʊ .~"
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
