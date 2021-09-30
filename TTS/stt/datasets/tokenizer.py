import re
from itertools import groupby
from typing import Dict, List


class Tokenizer:
    _CHARS_TO_IGNORE = "[\,\?\.\!\-\;\:\[\]\(\)'\`\"\“\”\’\\n]"

    def __init__(self, transcripts: List[str] = None, vocab_dict: Dict = None, chars_to_ignore: str = None):
        super().__init__()

        self.chars_to_ignore = chars_to_ignore if chars_to_ignore else self._CHARS_TO_IGNORE
        if transcripts:
            _, vocab_dict = self.extract_vocab(transcripts)
        self.encode_dict = vocab_dict
        self.decode_dict = {v: k for k, v in vocab_dict.items()}
        assert len(self.encode_dict) == len(self.decode_dict)

        self.unk_token = "[UNK]"
        self.pad_token = "[PAD]"
        self.word_del_token = "|"

        if self.unk_token not in self.encode_dict:
            self.encode_dict[self.unk_token] = len(self.encode_dict)
            self.decode_dict[len(self.decode_dict)] = self.unk_token

        if self.pad_token not in self.encode_dict:
            self.encode_dict[self.pad_token] = len(self.encode_dict)
            self.decode_dict[len(self.decode_dict)] = self.pad_token

        self.unk_token_id = self.encode_dict[self.unk_token]
        self.pad_token_id = self.encode_dict[self.pad_token]

    @property
    def vocab_size(self):
        return len(self.encode_dict)

    @property
    def vocab_dict(self):
        return self.encode_dict

    def encode(self, text: str):
        """Convert input text to sequence of ids."""
        text = text.lower()
        text = self.remove_special_characters(text)
        text = text.replace(" ", self.word_del_token)
        ids = [self.encode_dict.get(char, self.unk_token) for char in text]
        return ids

    def decode(self, token_ids: List[int]):
        """Convert input sequence of ids to text"""
        tokens = [self.decode_dict.get(ti, self.unk_token) for ti in token_ids]
        return tokens

    def tokens_to_string(self, tokens, group_tokens=True):
        # group same tokens into non-repeating tokens in CTC style decoding
        if group_tokens:
            tokens = [token_group[0] for token_group in groupby(tokens)]

        # filter self.pad_token which is used as CTC-blank token
        filtered_tokens = list(filter(lambda token: token != self.pad_token, tokens))

        # replace delimiter token
        string = "".join([" " if token == self.word_del_token else token for token in filtered_tokens]).strip()
        return string

    def remove_special_characters(self, text):
        text = re.sub(self.chars_to_ignore, "", text).lower()
        return text

    def extract_vocab(self, texts: List[str]):
        vocab = set()
        # for waveform, sr, _, text in dataset:
        for text in texts:
            text = text.lower()
            text = self.remove_special_characters(text)
            vocab.update(text)
        vocab = list(vocab)
        vocab_list = list(set(vocab))
        vocab_dict = {v: k for k, v in enumerate(vocab_list)}
        vocab_dict["|"] = vocab_dict[" "]
        del vocab_dict[" "]
        vocab_dict["[UNK]"] = len(vocab_dict)
        vocab_dict["[PAD]"] = len(vocab_dict)
        return vocab, vocab_dict
