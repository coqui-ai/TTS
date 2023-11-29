import os
import unittest

from trainer.io import save_checkpoint

from tests import get_tests_input_path
from TTS.config import load_config
from TTS.tts.models import setup_model
from TTS.utils.synthesizer import Synthesizer


class SynthesizerTest(unittest.TestCase):
    # pylint: disable=R0201
    def _create_random_model(self):
        # pylint: disable=global-statement
        config = load_config(os.path.join(get_tests_input_path(), "dummy_model_config.json"))
        model = setup_model(config)
        output_path = os.path.join(get_tests_input_path())
        save_checkpoint(config, model, None, None, 10, 1, output_path)

    def test_in_out(self):
        self._create_random_model()
        tts_root_path = get_tests_input_path()
        tts_checkpoint = os.path.join(tts_root_path, "checkpoint_10.pth")
        tts_config = os.path.join(tts_root_path, "dummy_model_config.json")
        synthesizer = Synthesizer(tts_checkpoint, tts_config, None, None)
        synthesizer.tts("Better this test works!!")

    def test_split_into_sentences(self):
        """Check demo server sentences split as expected"""
        print("\n > Testing demo server sentence splitting")
        # pylint: disable=attribute-defined-outside-init, protected-access
        self.seg = Synthesizer._get_segmenter("en")
        sis = Synthesizer.split_into_sentences
        assert sis(self, "Hello. Two sentences") == ["Hello.", "Two sentences"]
        assert sis(self, "He went to meet the adviser from Scott, Waltman & Co. next morning.") == [
            "He went to meet the adviser from Scott, Waltman & Co. next morning."
        ]
        assert sis(self, "Let's run it past Sarah and co. They'll want to see this.") == [
            "Let's run it past Sarah and co.",
            "They'll want to see this.",
        ]
        assert sis(self, "Where is Bobby Jr.'s rabbit?") == ["Where is Bobby Jr.'s rabbit?"]
        assert sis(self, "Please inform the U.K. authorities right away.") == [
            "Please inform the U.K. authorities right away."
        ]
        assert sis(self, "Were David and co. at the event?") == ["Were David and co. at the event?"]
        assert sis(self, "paging dr. green, please come to theatre four immediately.") == [
            "paging dr. green, please come to theatre four immediately."
        ]
        assert sis(self, "The email format is Firstname.Lastname@example.com. I think you reversed them.") == [
            "The email format is Firstname.Lastname@example.com.",
            "I think you reversed them.",
        ]
        assert sis(
            self,
            "The demo site is: https://top100.example.com/subsection/latestnews.html. Please send us your feedback.",
        ) == [
            "The demo site is: https://top100.example.com/subsection/latestnews.html.",
            "Please send us your feedback.",
        ]
        assert sis(self, "Scowling at him, 'You are not done yet!' she yelled.") == [
            "Scowling at him, 'You are not done yet!' she yelled."
        ]  # with the  final lowercase "she" we see it's all one sentence
        assert sis(self, "Hey!! So good to see you.") == ["Hey!!", "So good to see you."]
        assert sis(self, "He went to Yahoo! but I don't know the division.") == [
            "He went to Yahoo! but I don't know the division."
        ]
        assert sis(self, "If you can't remember a quote, “at least make up a memorable one that's plausible...\"") == [
            "If you can't remember a quote, “at least make up a memorable one that's plausible...\""
        ]
        assert sis(self, "The address is not google.com.") == ["The address is not google.com."]
        assert sis(self, "1.) The first item 2.) The second item") == ["1.) The first item", "2.) The second item"]
        assert sis(self, "1) The first item 2) The second item") == ["1) The first item", "2) The second item"]
        assert sis(self, "a. The first item b. The second item c. The third list item") == [
            "a. The first item",
            "b. The second item",
            "c. The third list item",
        ]
