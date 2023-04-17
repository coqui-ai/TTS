import unittest

from packaging.version import Version

from TTS.tts.utils.text.phonemizers import ESpeak, Gruut, JA_JP_Phonemizer, ZH_CN_Phonemizer
from TTS.tts.utils.text.phonemizers.bangla_phonemizer import BN_Phonemizer
from TTS.tts.utils.text.phonemizers.multi_phonemizer import MultiPhonemizer

EXAMPLE_TEXTs = [
    "Recent research at Harvard has shown meditating",
    "for as little as 8 weeks can actually increase, the grey matter",
    "in the parts of the brain responsible",
    "for emotional regulation and learning!",
]


EXPECTED_ESPEAK_PHONEMES = [
    "ɹ|ˈiː|s|ə|n|t ɹ|ɪ|s|ˈɜː|tʃ æ|t h|ˈɑːɹ|v|ɚ|d h|ɐ|z ʃ|ˈoʊ|n m|ˈɛ|d|ɪ|t|ˌeɪ|ɾ|ɪ|ŋ",
    "f|ɔː|ɹ æ|z l|ˈɪ|ɾ|əl æ|z ˈeɪ|t w|ˈiː|k|s k|æ|n ˈæ|k|tʃ|uː|əl|i| ˈɪ|n|k|ɹ|iː|s, ð|ə ɡ|ɹ|ˈeɪ m|ˈæ|ɾ|ɚ",
    "ɪ|n|ð|ə p|ˈɑːɹ|t|s ʌ|v|ð|ə b|ɹ|ˈeɪ|n ɹ|ɪ|s|p|ˈɑː|n|s|ə|b|əl",
    "f|ɔː|ɹ ɪ|m|ˈoʊ|ʃ|ə|n|əl ɹ|ˌɛ|ɡ|j|uː|l|ˈeɪ|ʃ|ə|n|| æ|n|d l|ˈɜː|n|ɪ|ŋ!",
]


EXPECTED_ESPEAK_v1_48_15_PHONEMES = [
    "ɹ|ˈiː|s|ə|n|t ɹ|ɪ|s|ˈɜː|tʃ æ|t h|ˈɑːɹ|v|ɚ|d h|ɐ|z ʃ|ˈoʊ|n m|ˈɛ|d|ᵻ|t|ˌeɪ|ɾ|ɪ|ŋ",
    "f|ɔː|ɹ æ|z l|ˈɪ|ɾ|əl æ|z ˈeɪ|t w|ˈiː|k|s k|æ|n ˈæ|k|tʃ|uː|əl|i| ˈɪ|n|k|ɹ|iː|s, ð|ə ɡ|ɹ|ˈeɪ m|ˈæ|ɾ|ɚ",
    "ɪ|n|ð|ə p|ˈɑːɹ|t|s ʌ|v|ð|ə b|ɹ|ˈeɪ|n ɹ|ɪ|s|p|ˈɑː|n|s|ə|b|əl",
    "f|ɔː|ɹ ɪ|m|ˈoʊ|ʃ|ə|n|əl ɹ|ˌɛ|ɡ|j|uː|l|ˈeɪ|ʃ|ə|n|| æ|n|d l|ˈɜː|n|ɪ|ŋ!",
]


EXPECTED_ESPEAKNG_PHONEMES = [
    "ɹ|ˈiː|s|ə|n|t ɹ|ᵻ|s|ˈɜː|tʃ æ|t h|ˈɑːɹ|v|ɚ|d h|ɐ|z ʃ|ˈoʊ|n m|ˈɛ|d|ᵻ|t|ˌeɪ|ɾ|ɪ|ŋ",
    "f|ɔː|ɹ æ|z l|ˈɪ|ɾ|əl æ|z ˈeɪ|t w|ˈiː|k|s k|æ|n ˈæ|k|tʃ|uː|əl|i| ˈɪ|ŋ|k|ɹ|iː|s, ð|ə ɡ|ɹ|ˈeɪ m|ˈæ|ɾ|ɚ",
    "ɪ|n|ð|ə p|ˈɑːɹ|t|s ʌ|v|ð|ə b|ɹ|ˈeɪ|n ɹ|ᵻ|s|p|ˈɑː|n|s|ᵻ|b|əl",
    "f|ɔː|ɹ ɪ|m|ˈoʊ|ʃ|ə|n|əl ɹ|ˌɛ|ɡ|j|ʊ|l|ˈeɪ|ʃ|ə|n|| æ|n|d l|ˈɜː|n|ɪ|ŋ!",
]


class TestEspeakPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = ESpeak(language="en-us", backend="espeak")

        if Version(self.phonemizer.backend_version) >= Version("1.48.15"):
            target_phonemes = EXPECTED_ESPEAK_v1_48_15_PHONEMES
        else:
            target_phonemes = EXPECTED_ESPEAK_PHONEMES

        for text, ph in zip(EXAMPLE_TEXTs, target_phonemes):
            phonemes = self.phonemizer.phonemize(text)
            self.assertEqual(phonemes, ph)

        # multiple punctuations
        text = "Be a voice, not an! echo?"
        gt = "biː ɐ vˈɔɪs, nˈɑːt ɐn! ˈɛkoʊ?"
        if Version(self.phonemizer.backend_version) >= Version("1.48.15"):
            gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ?"
        output = self.phonemizer.phonemize(text, separator="|")
        output = output.replace("|", "")
        self.assertEqual(output, gt)

        # not ending with punctuation
        text = "Be a voice, not an! echo"
        gt = "biː ɐ vˈɔɪs, nˈɑːt ɐn! ˈɛkoʊ"
        if Version(self.phonemizer.backend_version) >= Version("1.48.15"):
            gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ"
        output = self.phonemizer.phonemize(text, separator="")
        self.assertEqual(output, gt)

        # extra space after the sentence
        text = "Be a voice, not an! echo.  "
        gt = "biː ɐ vˈɔɪs, nˈɑːt ɐn! ˈɛkoʊ."
        if Version(self.phonemizer.backend_version) >= Version("1.48.15"):
            gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ."
        output = self.phonemizer.phonemize(text, separator="")
        self.assertEqual(output, gt)

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "espeak")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), dict)

    def test_get_version(self):
        self.assertIsInstance(self.phonemizer.version(), str)

    def test_is_available(self):
        self.assertTrue(self.phonemizer.is_available())


class TestEspeakNgPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = ESpeak(language="en-us", backend="espeak-ng")

        for text, ph in zip(EXAMPLE_TEXTs, EXPECTED_ESPEAKNG_PHONEMES):
            phonemes = self.phonemizer.phonemize(text)
            self.assertEqual(phonemes, ph)

        # multiple punctuations
        text = "Be a voice, not an! echo?"
        gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ?"
        output = self.phonemizer.phonemize(text, separator="|")
        output = output.replace("|", "")
        self.assertEqual(output, gt)

        # not ending with punctuation
        text = "Be a voice, not an! echo"
        gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ"
        output = self.phonemizer.phonemize(text, separator="")
        self.assertEqual(output, gt)

        # extra space after the sentence
        text = "Be a voice, not an! echo.  "
        gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ."
        output = self.phonemizer.phonemize(text, separator="")
        self.assertEqual(output, gt)

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "espeak")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), dict)

    def test_get_version(self):
        self.assertIsInstance(self.phonemizer.version(), str)

    def test_is_available(self):
        self.assertTrue(self.phonemizer.is_available())


class TestGruutPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = Gruut(language="en-us", use_espeak_phonemes=True, keep_stress=False)
        self.EXPECTED_PHONEMES = [
            "ɹ|i|ː|s|ə|n|t| ɹ|ᵻ|s|ɜ|ː|t|ʃ| æ|ɾ| h|ɑ|ː|ɹ|v|ɚ|d| h|ɐ|z| ʃ|o|ʊ|n| m|ɛ|d|ᵻ|t|e|ɪ|ɾ|ɪ|ŋ",
            "f|ɔ|ː|ɹ| æ|z| l|ɪ|ɾ|ə|l| æ|z| e|ɪ|t| w|i|ː|k|s| k|æ|ŋ| æ|k|t|ʃ|u|ː|ə|l|i| ɪ|ŋ|k|ɹ|i|ː|s, ð|ə| ɡ|ɹ|e|ɪ| m|æ|ɾ|ɚ",
            "ɪ|n| ð|ə| p|ɑ|ː|ɹ|t|s| ʌ|v| ð|ə| b|ɹ|e|ɪ|n| ɹ|ᵻ|s|p|ɑ|ː|n|s|ᵻ|b|ə|l",
            "f|ɔ|ː|ɹ| ɪ|m|o|ʊ|ʃ|ə|n|ə|l| ɹ|ɛ|ɡ|j|ʊ|l|e|ɪ|ʃ|ə|n| æ|n|d| l|ɜ|ː|n|ɪ|ŋ!",
        ]

    def test_phonemize(self):
        for text, ph in zip(EXAMPLE_TEXTs, self.EXPECTED_PHONEMES):
            phonemes = self.phonemizer.phonemize(text, separator="|")
            self.assertEqual(phonemes, ph)

        # multiple punctuations
        text = "Be a voice, not an! echo?"
        gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ?"
        output = self.phonemizer.phonemize(text, separator="|")
        output = output.replace("|", "")
        self.assertEqual(output, gt)

        # not ending with punctuation
        text = "Be a voice, not an! echo"
        gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ"
        output = self.phonemizer.phonemize(text, separator="")
        self.assertEqual(output, gt)

        # extra space after the sentence
        text = "Be a voice, not an! echo.  "
        gt = "biː ɐ vɔɪs, nɑːt ɐn! ɛkoʊ."
        output = self.phonemizer.phonemize(text, separator="")
        self.assertEqual(output, gt)

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "gruut")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), list)

    def test_get_version(self):
        self.assertIsInstance(self.phonemizer.version(), str)

    def test_is_available(self):
        self.assertTrue(self.phonemizer.is_available())


class TestJA_JPPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = JA_JP_Phonemizer()
        self._TEST_CASES = """
            どちらに行きますか？/dochiraniikimasuka?
            今日は温泉に、行きます。/kyo:waoNseNni,ikimasu.
            「A」から「Z」までです。/e:karazeqtomadedesu.
            そうですね！/so:desune!
            クジラは哺乳類です。/kujirawahonyu:ruidesu.
            ヴィディオを見ます。/bidioomimasu.
            今日は８月22日です/kyo:wahachigatsuniju:ninichidesu
            xyzとαβγ/eqkusuwaizeqtotoarufabe:tagaNma
            値段は$12.34です/nedaNwaju:niteNsaNyoNdorudesu
            """

    def test_phonemize(self):
        for line in self._TEST_CASES.strip().split("\n"):
            text, phone = line.split("/")
            self.assertEqual(self.phonemizer.phonemize(text, separator=""), phone)

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "ja_jp_phonemizer")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), dict)

    def test_get_version(self):
        self.assertIsInstance(self.phonemizer.version(), str)

    def test_is_available(self):
        self.assertTrue(self.phonemizer.is_available())


class TestZH_CN_Phonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = ZH_CN_Phonemizer()
        self._TEST_CASES = ""

    def test_phonemize(self):
        # TODO: implement ZH phonemizer tests
        pass

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "zh_cn_phonemizer")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), dict)

    def test_get_version(self):
        self.assertIsInstance(self.phonemizer.version(), str)

    def test_is_available(self):
        self.assertTrue(self.phonemizer.is_available())


class TestBN_Phonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = BN_Phonemizer()
        self._TEST_CASES = "রাসূলুল্লাহ সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম শিক্ষা দিয়েছেন যে, কেউ যদি কোন খারাপ কিছুর সম্মুখীন হয়, তখনও যেন"
        self._EXPECTED = "রাসূলুল্লাহ সাল্লাল্লাহু আলাইহি ওয়া সাল্লাম শিক্ষা দিয়েছেন যে কেউ যদি কোন খারাপ কিছুর সম্মুখীন হয় তখনও যেন।"

    def test_phonemize(self):
        self.assertEqual(self.phonemizer.phonemize(self._TEST_CASES, separator=""), self._EXPECTED)

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "bn_phonemizer")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), dict)

    def test_get_version(self):
        self.assertIsInstance(self.phonemizer.version(), str)

    def test_is_available(self):
        self.assertTrue(self.phonemizer.is_available())


class TestMultiPhonemizer(unittest.TestCase):
    def setUp(self):
        self.phonemizer = MultiPhonemizer({"tr": "espeak", "en-us": "", "de": "gruut", "zh-cn": ""})

    def test_phonemize(self):
        # Enlish espeak
        text = "Be a voice, not an! echo?"
        gt = "biː ɐ vˈɔɪs, nˈɑːt æn! ˈɛkoʊ?"
        output = self.phonemizer.phonemize(text, separator="|", language="en-us")
        output = output.replace("|", "")
        self.assertEqual(output, gt)

        # German gruut
        text = "Hallo, das ist ein Deutches Beipiel!"
        gt = "haloː, das ɪst aeːn dɔɔʏ̯tçəs bəʔiːpiːl!"
        output = self.phonemizer.phonemize(text, separator="|", language="de")
        output = output.replace("|", "")
        self.assertEqual(output, gt)

    def test_phonemizer_initialization(self):
        # test with unsupported language
        with self.assertRaises(ValueError):
            MultiPhonemizer({"tr": "espeak", "xx": ""})

        # test with unsupported phonemizer
        with self.assertRaises(ValueError):
            MultiPhonemizer({"tr": "espeak", "fr": "xx"})

    def test_sub_phonemizers(self):
        for lang in self.phonemizer.lang_to_phonemizer_name.keys():
            self.assertEqual(lang, self.phonemizer.lang_to_phonemizer[lang].language)
            self.assertEqual(
                self.phonemizer.lang_to_phonemizer_name[lang], self.phonemizer.lang_to_phonemizer[lang].name()
            )

    def test_name(self):
        self.assertEqual(self.phonemizer.name(), "multi-phonemizer")

    def test_get_supported_languages(self):
        self.assertIsInstance(self.phonemizer.supported_languages(), list)
