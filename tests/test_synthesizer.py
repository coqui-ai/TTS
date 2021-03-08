import os
import unittest

from tests import get_tests_input_path, get_tests_output_path
from TTS.utils.synthesizer import Synthesizer
from TTS.tts.utils.generic_utils import setup_model
from TTS.tts.utils.io import save_checkpoint
from TTS.tts.utils.text.symbols import make_symbols, phonemes, symbols
from TTS.utils.io import load_config


class SynthesizerTest(unittest.TestCase):
    # pylint: disable=R0201
    def _create_random_model(self):
        # pylint: disable=global-statement
        global symbols, phonemes
        config = load_config(os.path.join(get_tests_output_path(), 'dummy_model_config.json'))
        if 'characters' in config.keys():
            symbols, phonemes = make_symbols(**config.characters)

        num_chars = len(phonemes) if config.use_phonemes else len(symbols)
        model = setup_model(num_chars, 0, config)
        output_path = os.path.join(get_tests_output_path())
        save_checkpoint(model, None, 10, 10, 1, output_path, None)

    def test_in_out(self):
        self._create_random_model()
        config = load_config(os.path.join(get_tests_input_path(), 'server_config.json'))
        tts_root_path = get_tests_output_path()
        config['tts_checkpoint'] = os.path.join(tts_root_path, config['tts_checkpoint'])
        config['tts_config'] = os.path.join(tts_root_path, config['tts_config'])
        synthesizer = Synthesizer(config['tts_checkpoint'], config['tts_config'], None, None)
        synthesizer.tts("Better this test works!!")

    def test_split_into_sentences(self):
        """Check demo server sentences split as expected"""
        print("\n > Testing demo server sentence splitting")
        # pylint: disable=attribute-defined-outside-init
        self.seg = Synthesizer.get_segmenter("en")
        sis = Synthesizer.split_into_sentences
        assert sis(self, 'Hello. Two sentences') == ['Hello.', 'Two sentences']
        assert sis(self, 'He went to meet the adviser from Scott, Waltman & Co. next morning.') == ['He went to meet the adviser from Scott, Waltman & Co. next morning.']
        assert sis(self, 'Let\'s run it past Sarah and co. They\'ll want to see this.') == ['Let\'s run it past Sarah and co.', 'They\'ll want to see this.']
        assert sis(self, 'Where is Bobby Jr.\'s rabbit?') == ['Where is Bobby Jr.\'s rabbit?']
        assert sis(self, 'Please inform the U.K. authorities right away.') == ['Please inform the U.K. authorities right away.']
        assert sis(self, 'Were David and co. at the event?') == ['Were David and co. at the event?']
        assert sis(self, 'paging dr. green, please come to theatre four immediately.') == ['paging dr. green, please come to theatre four immediately.']
        assert sis(self, 'The email format is Firstname.Lastname@example.com. I think you reversed them.') == ['The email format is Firstname.Lastname@example.com.', 'I think you reversed them.']
        assert sis(self, 'The demo site is: https://top100.example.com/subsection/latestnews.html. Please send us your feedback.') == ['The demo site is: https://top100.example.com/subsection/latestnews.html.', 'Please send us your feedback.']
        assert sis(self, 'Scowling at him, \'You are not done yet!\' she yelled.') == ['Scowling at him, \'You are not done yet!\' she yelled.'] # with the  final lowercase "she" we see it's all one sentence
        assert sis(self, 'Hey!! So good to see you.') == ['Hey!!', 'So good to see you.']
        assert sis(self, 'He went to Yahoo! but I don\'t know the division.') == ['He went to Yahoo! but I don\'t know the division.']
        assert sis(self, 'If you can\'t remember a quote, “at least make up a memorable one that\'s plausible..."') == ['If you can\'t remember a quote, “at least make up a memorable one that\'s plausible..."']
        assert sis(self, 'The address is not google.com.') == ['The address is not google.com.']
        assert sis(self, '1.) The first item 2.) The second item') == ['1.) The first item', '2.) The second item']
        assert sis(self, '1) The first item 2) The second item') == ['1) The first item', '2) The second item']
        assert sis(self, 'a. The first item b. The second item c. The third list item') == ['a. The first item', 'b. The second item', 'c. The third list item']
