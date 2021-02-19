from TTS.utils.io import AttrDict
from torch import nn
from abc import ABC, abstractmethod


class VocoderAbstract(ABC, nn.Module):
    """Abstract for vocoder model (melgan, wavernn, etc ...)

    Heritance:
        ABC: Abstract Base Class
        nn.Module: pytorch nn.Module
    """
    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inference(self, text, speaker_ids=None, style_mel=None, speaker_embeddings=None):
        pass

    @abstractmethod
    def load_checkpoint(self, config: AttrDict, checkpoint_path: str, eval: bool = False):
        pass


