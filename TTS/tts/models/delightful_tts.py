from json import encoder
import math
import os
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.modules.conv as conv
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import BaseTTSConfig
from TTS.tts.datasets.dataset import F0Dataset, TTSDataset, _parse_sample
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.layers.losses import ForwardSumLoss, SSIMLoss, VitsDiscriminatorLoss
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.models.base_tts import BaseTTSE2E
from TTS.tts.models.vits import load_audio, wav_to_energy, wav_to_mel
from TTS.tts.utils.emotions import EmotionManager
from TTS.tts.utils.helpers import (
    average_over_durations,
    compute_attn_prior,
    generate_path,
    maximum_path,
    rand_segments,
    segment,
    sequence_mask,
)
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_avg_pitch, plot_pitch, plot_spectrogram
from TTS.utils.audio.numpy_transforms import build_mel_basis, compute_f0
from TTS.utils.audio.numpy_transforms import db_to_amp as db_to_amp_numpy
from TTS.utils.audio.numpy_transforms import mel_to_wav as mel_to_wav_numpy
from TTS.utils.io import load_fsspec
from TTS.vocoder.layers.losses import MultiScaleSTFTLoss
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

#############################
# CONFIGS
#############################

@dataclass
class ConformerConfig(Coqpit):
    n_layers: int
    n_heads: int
    n_hidden: int
    p_dropout: float
    kernel_size_conv_mod: int
    kernel_size_depthwise: int


@dataclass
class ReferenceEncoderConfig(Coqpit):
    bottleneck_size_p: int
    bottleneck_size_u: int
    ref_enc_filters: List[int]
    ref_enc_size: int
    ref_enc_strides: List[int]
    ref_enc_pad: List[int]
    ref_enc_gru_size: int
    ref_attention_dropout: float
    token_num: int
    predictor_kernel_size: int


@dataclass
class VarianceAdaptorConfig(Coqpit):
    n_hidden: int
    kernel_size: int
    emb_kernel_size: int
    p_dropout: float
    n_bins: int


@dataclass
class AcousticModelConfig(Coqpit):
    encoder: ConformerConfig = ConformerConfig(
        n_layers=6,
        n_heads=8,
        n_hidden=512,
        p_dropout=0.1,
        kernel_size_conv_mod=7,
        kernel_size_depthwise=7,
    )
    decoder: ConformerConfig = ConformerConfig(
        n_layers=6,
        n_heads=8,
        n_hidden=512,
        p_dropout=0.1,
        kernel_size_conv_mod=11,
        kernel_size_depthwise=11,
    )
    reference_encoder: ReferenceEncoderConfig = ReferenceEncoderConfig(
        bottleneck_size_p=4,
        bottleneck_size_u=256,
        ref_enc_filters=[32, 32, 64, 64, 128, 128],
        ref_enc_size=3,
        ref_enc_strides=[1, 2, 1, 2, 1],
        ref_enc_pad=[1, 1],
        ref_enc_gru_size=32,
        ref_attention_dropout=0.2,
        token_num=32,
        predictor_kernel_size=5,
    )
    variance_adaptor: VarianceAdaptorConfig = VarianceAdaptorConfig(
        n_hidden=512, kernel_size=5, p_dropout=0.5, n_bins=256, emb_kernel_size=3
    )


@dataclass
class DelightfulTtsAudioConfig(Coqpit):
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    fft_size: int = 1024
    mel_fmin: float = 0.0
    mel_fmax: float = 8000
    num_mels: int = 100
    pitch_fmax: float = 640.0


##############################
# MODEL DEFINITION
##############################

@dataclass
class DelightfulTtsArgs(Coqpit):
    num_chars: int = 100
    spec_segment_size: int = 32
    # discriminator
    acoustic_config: AcousticModelConfig = AcousticModelConfig()
    # multi-speaker params
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 384
    use_d_vector_file: bool = False
    d_vector_dim: int = 0
    use_emotion_embedding: bool = False
    use_emotion_vector_file: bool = False
    emotion_vector_dim: int = 0
    emotion_vector_file: str = None
    # freeze layers
    freeze_vocoder: bool = False
    freeze_text_encoder: bool = False
    freeze_duration_predictor: bool = False
    freeze_pitch_predictor: bool = False
    freeze_energy_predictor: bool = False
    freeze_basis_vectors_predictor: bool = False
    freeze_decoder: bool = False
    # aux params
    length_scale: float = 1.0