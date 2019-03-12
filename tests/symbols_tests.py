import unittest

from utils.text import phonemes

class SymbolsTest(unittest.TestCase):
    def test_uniqueness(self):
        assert sorted(phonemes) == sorted(list(set(phonemes)))
