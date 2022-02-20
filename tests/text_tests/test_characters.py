import unittest

from TTS.tts.utils.text.characters import BaseCharacters, Graphemes, IPAPhonemes

# pylint: disable=protected-access


class BaseCharacterTest(unittest.TestCase):
    def setUp(self):
        self.characters_empty = BaseCharacters("", "", pad="", eos="", bos="", blank="", is_unique=True, is_sorted=True)

    def test_default_character_sets(self):  # pylint: disable=no-self-use
        """Test initiation of default character sets"""
        _ = IPAPhonemes()
        _ = Graphemes()

    def test_unique(self):
        """Test if the unique option works"""
        self.characters_empty.characters = "abcc"
        self.characters_empty.punctuations = ".,;:!? "
        self.characters_empty.pad = "[PAD]"
        self.characters_empty.eos = "[EOS]"
        self.characters_empty.bos = "[BOS]"
        self.characters_empty.blank = "[BLANK]"

        self.assertEqual(
            self.characters_empty.num_chars,
            len(["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]),
        )

    def test_unique_sorted(self):
        """Test if the unique and sorted option works"""
        self.characters_empty.characters = "cba"
        self.characters_empty.punctuations = ".,;:!? "
        self.characters_empty.pad = "[PAD]"
        self.characters_empty.eos = "[EOS]"
        self.characters_empty.bos = "[BOS]"
        self.characters_empty.blank = "[BLANK]"

        self.assertEqual(
            self.characters_empty.num_chars,
            len(["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]),
        )

    def test_setters_getters(self):
        """Test the class setters behaves as expected"""
        self.characters_empty.characters = "abc"
        self.assertEqual(self.characters_empty._characters, "abc")
        self.assertEqual(self.characters_empty.vocab, ["a", "b", "c"])

        self.characters_empty.punctuations = ".,;:!? "
        self.assertEqual(self.characters_empty._punctuations, ".,;:!? ")
        self.assertEqual(self.characters_empty.vocab, ["a", "b", "c", ".", ",", ";", ":", "!", "?", " "])

        self.characters_empty.pad = "[PAD]"
        self.assertEqual(self.characters_empty._pad, "[PAD]")
        self.assertEqual(self.characters_empty.vocab, ["[PAD]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "])

        self.characters_empty.eos = "[EOS]"
        self.assertEqual(self.characters_empty._eos, "[EOS]")
        self.assertEqual(
            self.characters_empty.vocab, ["[PAD]", "[EOS]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]
        )

        self.characters_empty.bos = "[BOS]"
        self.assertEqual(self.characters_empty._bos, "[BOS]")
        self.assertEqual(
            self.characters_empty.vocab, ["[PAD]", "[EOS]", "[BOS]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]
        )

        self.characters_empty.blank = "[BLANK]"
        self.assertEqual(self.characters_empty._blank, "[BLANK]")
        self.assertEqual(
            self.characters_empty.vocab,
            ["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "],
        )
        self.assertEqual(
            self.characters_empty.num_chars,
            len(["[PAD]", "[EOS]", "[BOS]", "[BLANK]", "a", "b", "c", ".", ",", ";", ":", "!", "?", " "]),
        )

        self.characters_empty.print_log()

    def test_char_lookup(self):
        """Test char to ID and ID to char conversion"""
        self.characters_empty.characters = "abc"
        self.characters_empty.punctuations = ".,;:!? "
        self.characters_empty.pad = "[PAD]"
        self.characters_empty.eos = "[EOS]"
        self.characters_empty.bos = "[BOS]"
        self.characters_empty.blank = "[BLANK]"

        # char to ID
        self.assertEqual(self.characters_empty.char_to_id("[PAD]"), 0)
        self.assertEqual(self.characters_empty.char_to_id("[EOS]"), 1)
        self.assertEqual(self.characters_empty.char_to_id("[BOS]"), 2)
        self.assertEqual(self.characters_empty.char_to_id("[BLANK]"), 3)
        self.assertEqual(self.characters_empty.char_to_id("a"), 4)
        self.assertEqual(self.characters_empty.char_to_id("b"), 5)
        self.assertEqual(self.characters_empty.char_to_id("c"), 6)
        self.assertEqual(self.characters_empty.char_to_id("."), 7)
        self.assertEqual(self.characters_empty.char_to_id(","), 8)
        self.assertEqual(self.characters_empty.char_to_id(";"), 9)
        self.assertEqual(self.characters_empty.char_to_id(":"), 10)
        self.assertEqual(self.characters_empty.char_to_id("!"), 11)
        self.assertEqual(self.characters_empty.char_to_id("?"), 12)
        self.assertEqual(self.characters_empty.char_to_id(" "), 13)

        # ID to char
        self.assertEqual(self.characters_empty.id_to_char(0), "[PAD]")
        self.assertEqual(self.characters_empty.id_to_char(1), "[EOS]")
        self.assertEqual(self.characters_empty.id_to_char(2), "[BOS]")
        self.assertEqual(self.characters_empty.id_to_char(3), "[BLANK]")
        self.assertEqual(self.characters_empty.id_to_char(4), "a")
        self.assertEqual(self.characters_empty.id_to_char(5), "b")
        self.assertEqual(self.characters_empty.id_to_char(6), "c")
        self.assertEqual(self.characters_empty.id_to_char(7), ".")
        self.assertEqual(self.characters_empty.id_to_char(8), ",")
        self.assertEqual(self.characters_empty.id_to_char(9), ";")
        self.assertEqual(self.characters_empty.id_to_char(10), ":")
        self.assertEqual(self.characters_empty.id_to_char(11), "!")
        self.assertEqual(self.characters_empty.id_to_char(12), "?")
        self.assertEqual(self.characters_empty.id_to_char(13), " ")
