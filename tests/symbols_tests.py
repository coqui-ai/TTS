import unittest

from utils.text import phonemes
from collections import Counter

class SymbolsTest(unittest.TestCase):
    def test_uniqueness(self):
        assert sorted(phonemes) == sorted(list(set(phonemes))), " {} vs {} ".format(len(phonemes), len(set(phonemes)))
        