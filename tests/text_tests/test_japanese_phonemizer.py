import unittest

from TTS.tts.utils.text.japanese.phonemizer import japanese_text_to_phonemes

_TEST_CASES = """
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


class TestText(unittest.TestCase):
    def test_japanese_text_to_phonemes(self):
        for line in _TEST_CASES.strip().split("\n"):
            text, phone = line.split("/")
            self.assertEqual(japanese_text_to_phonemes(text), phone)


if __name__ == "__main__":
    unittest.main()
