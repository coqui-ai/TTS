from typing import Callable, Dict, List, Union

from TTS.tts.utils.text import cleaners
from TTS.tts.utils.text.characters import Graphemes, IPAPhonemes
from TTS.tts.utils.text.phonemizers import DEF_LANG_TO_PHONEMIZER, get_phonemizer_by_name
from TTS.tts.utils.text.phonemizers.multi_phonemizer import MultiPhonemizer
from TTS.utils.generic_utils import get_import_path, import_class


class TTSTokenizer:
    """🐸TTS tokenizer to convert input characters to token IDs and back.

    Token IDs for OOV chars are discarded but those are stored in `self.not_found_characters` for later.

    Args:
        use_phonemes (bool):
            Whether to use phonemes instead of characters. Defaults to False.

        characters (Characters):
            A Characters object to use for character-to-ID and ID-to-character mappings.

        text_cleaner (callable):
            A function to pre-process the text before tokenization and phonemization. Defaults to None.

        phonemizer (Phonemizer):
            A phonemizer object or a dict that maps language codes to phonemizer objects. Defaults to None.

    Example:

        >>> from TTS.tts.utils.text.tokenizer import TTSTokenizer
        >>> tokenizer = TTSTokenizer(use_phonemes=False, characters=Graphemes())
        >>> text = "Hello world!"
        >>> ids = tokenizer.text_to_ids(text)
        >>> text_hat = tokenizer.ids_to_text(ids)
        >>> assert text == text_hat
    """

    def __init__(
        self,
        use_phonemes=False,
        text_cleaner: Callable = None,
        characters: "BaseCharacters" = None,
        phonemizer: Union["Phonemizer", Dict] = None,
        add_blank: bool = False,
        use_eos_bos=False,
    ):
        self.text_cleaner = text_cleaner
        self.use_phonemes = use_phonemes
        self.add_blank = add_blank
        self.use_eos_bos = use_eos_bos
        self.characters = characters
        self.not_found_characters = []
        self.phonemizer = phonemizer

    @property
    def characters(self):
        return self._characters

    @characters.setter
    def characters(self, new_characters):
        self._characters = new_characters
        self.pad_id = self.characters.char_to_id(self.characters.pad) if self.characters.pad else None
        self.blank_id = self.characters.char_to_id(self.characters.blank) if self.characters.blank else None

    def encode(self, text: str) -> List[int]:
        """Encodes a string of text as a sequence of IDs."""
        token_ids = []
        for char in text:
            try:
                idx = self.characters.char_to_id(char)
                token_ids.append(idx)
            except KeyError:
                # discard but store not found characters
                if char not in self.not_found_characters:
                    self.not_found_characters.append(char)
                    print(text)
                    print(f" [!] Character {repr(char)} not found in the vocabulary. Discarding it.")
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes a sequence of IDs to a string of text."""
        text = ""
        for token_id in token_ids:
            text += self.characters.id_to_char(token_id)
        return text

    def text_to_ids(self, text: str, language: str = None) -> List[int]:  # pylint: disable=unused-argument
        """Converts a string of text to a sequence of token IDs.

        Args:
            text(str):
                The text to convert to token IDs.

            language(str):
                The language code of the text. Defaults to None.

        TODO:
            - Add support for language-specific processing.

        1. Text normalizatin
        2. Phonemization (if use_phonemes is True)
        3. Add blank char between characters
        4. Add BOS and EOS characters
        5. Text to token IDs
        """
        # TODO: text cleaner should pick the right routine based on the language
        if self.text_cleaner is not None:
            text = self.text_cleaner(text)
        if self.use_phonemes:
            text = self.phonemizer.phonemize(text, separator="", language=language)
        text = self.encode(text)
        if self.add_blank:
            text = self.intersperse_blank_char(text, True)
        if self.use_eos_bos:
            text = self.pad_with_bos_eos(text)
        return text

    def ids_to_text(self, id_sequence: List[int]) -> str:
        """Converts a sequence of token IDs to a string of text."""
        return self.decode(id_sequence)

    def pad_with_bos_eos(self, char_sequence: List[str]):
        """Pads a sequence with the special BOS and EOS characters."""
        return [self.characters.bos_id] + list(char_sequence) + [self.characters.eos_id]

    def intersperse_blank_char(self, char_sequence: List[str], use_blank_char: bool = False):
        """Intersperses the blank character between characters in a sequence.

        Use the ```blank``` character if defined else use the ```pad``` character.
        """
        char_to_use = self.characters.blank_id if use_blank_char else self.characters.pad
        result = [char_to_use] * (len(char_sequence) * 2 + 1)
        result[1::2] = char_sequence
        return result

    def print_logs(self, level: int = 0):
        indent = "\t" * level
        print(f"{indent}| > add_blank: {self.add_blank}")
        print(f"{indent}| > use_eos_bos: {self.use_eos_bos}")
        print(f"{indent}| > use_phonemes: {self.use_phonemes}")
        if self.use_phonemes:
            print(f"{indent}| > phonemizer:")
            self.phonemizer.print_logs(level + 1)
        if len(self.not_found_characters) > 0:
            print(f"{indent}| > {len(self.not_found_characters)} not found characters:")
            for char in self.not_found_characters:
                print(f"{indent}| > {char}")

    @staticmethod
    def init_from_config(config: "Coqpit", characters: "BaseCharacters" = None):
        """Init Tokenizer object from config

        Args:
            config (Coqpit): Coqpit model config.
            characters (BaseCharacters): Defines the model character set. If not set, use the default options based on
                the config values. Defaults to None.
        """
        # init cleaners
        text_cleaner = None
        if isinstance(config.text_cleaner, (str, list)):
            text_cleaner = getattr(cleaners, config.text_cleaner)

        # init characters
        if characters is None:
            # set characters based on defined characters class
            if config.characters and config.characters.characters_class:
                CharactersClass = import_class(config.characters.characters_class)
                characters, new_config = CharactersClass.init_from_config(config)
            # set characters based on config
            else:
                if config.use_phonemes:
                    # init phoneme set
                    characters, new_config = IPAPhonemes().init_from_config(config)
                else:
                    # init character set
                    characters, new_config = Graphemes().init_from_config(config)

        else:
            characters, new_config = characters.init_from_config(config)

        # set characters class
        new_config.characters.characters_class = get_import_path(characters)

        # init phonemizer
        phonemizer = None
        if config.use_phonemes:
            if "phonemizer" in config and config.phonemizer == "multi_phonemizer":
                lang_to_phonemizer_name = {}
                for dataset in config.datasets:
                    if dataset.language != "":
                        lang_to_phonemizer_name[dataset.language] = dataset.phonemizer
                    else:
                        raise ValueError("Multi phonemizer requires language to be set for each dataset.")
                phonemizer = MultiPhonemizer(lang_to_phonemizer_name)
            else:
                phonemizer_kwargs = {"language": config.phoneme_language}
                if "phonemizer" in config and config.phonemizer:
                    phonemizer = get_phonemizer_by_name(config.phonemizer, **phonemizer_kwargs)
                else:
                    try:
                        phonemizer = get_phonemizer_by_name(
                            DEF_LANG_TO_PHONEMIZER[config.phoneme_language], **phonemizer_kwargs
                        )
                        new_config.phonemizer = phonemizer.name()
                    except KeyError as e:
                        raise ValueError(
                            f"""No phonemizer found for language {config.phoneme_language}.
                            You may need to install a third party library for this language."""
                        ) from e

        return (
            TTSTokenizer(
                config.use_phonemes, text_cleaner, characters, phonemizer, config.add_blank, config.enable_eos_bos_chars
            ),
            new_config,
        )
