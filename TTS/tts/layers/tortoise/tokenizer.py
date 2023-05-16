import os

import torch
from tokenizers import Tokenizer

from TTS.tts.utils.text.cleaners import english_cleaners

DEFAULT_VOCAB_FILE = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "../../utils/assets/tortoise/tokenizer.json"
)


class VoiceBpeTokenizer:
    def __init__(self, vocab_file=DEFAULT_VOCAB_FILE):
        if vocab_file is not None:
            self.tokenizer = Tokenizer.from_file(vocab_file)

    def preprocess_text(self, txt):
        txt = english_cleaners(txt)
        return txt

    def encode(self, txt):
        txt = self.preprocess_text(txt)
        txt = txt.replace(" ", "[SPACE]")
        return self.tokenizer.encode(txt).ids

    def decode(self, seq):
        if isinstance(seq, torch.Tensor):
            seq = seq.cpu().numpy()
        txt = self.tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace("[SPACE]", " ")
        txt = txt.replace("[STOP]", "")
        txt = txt.replace("[UNK]", "")
        return txt
