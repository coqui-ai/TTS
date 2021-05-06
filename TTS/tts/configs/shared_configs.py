from dataclasses import asdict, dataclass, field
from typing import List

from coqpit import MISSING, Coqpit, check_argument

from TTS.config import BaseAudioConfig, BaseDatasetConfig, BaseTrainingConfig


@dataclass
class GSTConfig(Coqpit):
    """Defines Global Style Toke module"""

    gst_style_input_wav: str = None
    gst_style_input_weights: dict = None
    gst_embedding_dim: int = 256
    gst_use_speaker_embedding: bool = False
    gst_num_heads: int = 4
    gst_num_style_tokens: int = 10

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        super().check_values()
        check_argument("gst_style_input_weights", c, restricted=False)
        check_argument("gst_style_input_wav", c, restricted=False)
        check_argument("gst_embedding_dim", c, restricted=True, min_val=0, max_val=1000)
        check_argument("gst_use_speaker_embedding", c, restricted=False)
        check_argument("gst_num_heads", c, restricted=True, min_val=2, max_val=10)
        check_argument("gst_num_style_tokens", c, restricted=True, min_val=1, max_val=1000)


@dataclass
class CharactersConfig:
    """Defines character or phoneme set used by the model"""

    pad: str = None
    eos: str = None
    bos: str = None
    characters: str = None
    punctuations: str = None
    phonemes: str = None

    def check_values(
        self,
    ):
        """Check config fields"""
        c = asdict(self)
        check_argument("pad", c, "characters", restricted=True)
        check_argument("eos", c, "characters", restricted=True)
        check_argument("bos", c, "characters", restricted=True)
        check_argument("characters", c, "characters", restricted=True)
        check_argument("phonemes", c, restricted=True)
        check_argument("punctuations", c, "characters", restricted=True)


@dataclass
class BaseTTSConfig(BaseTrainingConfig):
    """Shared parameters among all the tts models."""

    audio: BaseAudioConfig = field(default_factory=BaseAudioConfig)
    # phoneme settings
    use_phonemes: bool = False
    phoneme_language: str = None
    compute_input_seq_cache: bool = False
    text_cleaner: str = MISSING
    enable_eos_bos_chars: bool = False
    test_sentences_file: str = ""
    phoneme_cache_path: str = None
    # vocabulary parameters
    characters: CharactersConfig = None
    # training params
    batch_group_size: int = 0
    loss_masking: bool = None
    # dataloading
    min_seq_len: int = 1
    max_seq_len: int = float("inf")
    compute_f0: bool = False
    use_noise_augment: bool = False
    add_blank: bool = False
    # dataset
    datasets: List[BaseDatasetConfig] = field(default_factory=lambda: [BaseDatasetConfig()])
