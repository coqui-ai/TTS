import unittest
from dataclasses import dataclass, field

from coqpit import Coqpit

from TTS.tts.utils.text.characters import Graphemes, IPAPhonemes, _blank, _bos, _eos, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.phonemizers import ESpeak
from TTS.tts.utils.text.tokenizer import TTSTokenizer


class TestTTSTokenizer(unittest.TestCase):
    def setUp(self):
        self.tokenizer = TTSTokenizer(use_phonemes=False, characters=Graphemes())

        self.ph = ESpeak("tr", backend="espeak")
        self.tokenizer_ph = TTSTokenizer(use_phonemes=True, characters=IPAPhonemes(), phonemizer=self.ph)

    def test_encode_decode_graphemes(self):
        text = "This is, a test."
        ids = self.tokenizer.encode(text)
        test_hat = self.tokenizer.decode(ids)
        self.assertEqual(text, test_hat)
        self.assertEqual(len(ids), len(text))

    def test_text_to_ids_phonemes(self):
        # TODO: note sure how to extend to cover all the languages and phonemizer.
        text = "Bu bir Örnek."
        text_ph = self.ph.phonemize(text, separator="")
        ids = self.tokenizer_ph.text_to_ids(text)
        test_hat = self.tokenizer_ph.ids_to_text(ids)
        self.assertEqual(text_ph, test_hat)

    def test_text_to_ids_phonemes_punctuation(self):
        text = "..."
        text_ph = self.ph.phonemize(text, separator="")
        ids = self.tokenizer_ph.text_to_ids(text)
        test_hat = self.tokenizer_ph.ids_to_text(ids)
        self.assertEqual(text_ph, test_hat)

    def test_text_to_ids_phonemes_with_eos_bos(self):
        text = "Bu bir Örnek."
        self.tokenizer_ph.use_eos_bos = True
        text_ph = IPAPhonemes().bos + self.ph.phonemize(text, separator="") + IPAPhonemes().eos
        ids = self.tokenizer_ph.text_to_ids(text)
        test_hat = self.tokenizer_ph.ids_to_text(ids)
        self.assertEqual(text_ph, test_hat)

    def test_text_to_ids_phonemes_with_eos_bos_and_blank(self):
        text = "Bu bir Örnek."
        self.tokenizer_ph.use_eos_bos = True
        self.tokenizer_ph.add_blank = True
        text_ph = "<BOS><BLNK>b<BLNK>ʊ<BLNK> <BLNK>b<BLNK>ɪ<BLNK>r<BLNK> <BLNK>œ<BLNK>r<BLNK>n<BLNK>ˈ<BLNK>ɛ<BLNK>c<BLNK>.<BLNK><EOS>"
        ids = self.tokenizer_ph.text_to_ids(text)
        text_hat = self.tokenizer_ph.ids_to_text(ids)
        self.assertEqual(text_ph, text_hat)

    def test_print_logs(self):
        self.tokenizer.print_logs()
        self.tokenizer_ph.print_logs()

    def test_not_found_characters(self):
        self.ph = ESpeak("en-us")
        tokenizer_local = TTSTokenizer(use_phonemes=True, characters=IPAPhonemes(), phonemizer=self.ph)
        self.assertEqual(len(self.tokenizer.not_found_characters), 0)
        text = "Yolk of one egg beaten light"
        ids = tokenizer_local.text_to_ids(text)
        text_hat = tokenizer_local.ids_to_text(ids)
        self.assertEqual(tokenizer_local.not_found_characters, ["̩"])
        self.assertEqual(text_hat, "jˈoʊk ʌv wˈʌn ˈɛɡ bˈiːʔn lˈaɪt")

    def test_init_from_config(self):
        @dataclass
        class Characters(Coqpit):
            characters_class: str = None
            characters: str = _phonemes
            punctuations: str = _punctuations
            pad: str = _pad
            eos: str = _eos
            bos: str = _bos
            blank: str = _blank
            is_unique: bool = True
            is_sorted: bool = True

        @dataclass
        class TokenizerConfig(Coqpit):
            enable_eos_bos_chars: bool = True
            use_phonemes: bool = True
            add_blank: bool = False
            characters: str = field(default_factory=Characters)
            phonemizer: str = "espeak"
            phoneme_language: str = "tr"
            text_cleaner: str = "phoneme_cleaners"
            characters = field(default_factory=Characters)

        tokenizer_ph, _ = TTSTokenizer.init_from_config(TokenizerConfig())
        tokenizer_ph.phonemizer.backend = "espeak"
        text = "Bu bir Örnek."
        text_ph = "<BOS>" + self.ph.phonemize(text, separator="") + "<EOS>"
        ids = tokenizer_ph.text_to_ids(text)
        test_hat = tokenizer_ph.ids_to_text(ids)
        self.assertEqual(text_ph, test_hat)
