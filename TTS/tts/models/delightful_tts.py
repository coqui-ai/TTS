import os
from dataclasses import dataclass, field
from itertools import chain
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.datasets.dataset import F0Dataset, TTSDataset, _parse_sample
from TTS.tts.layers.delightful_tts.acoustic_model import AcousticModel
from TTS.tts.layers.losses import ForwardSumLoss, SSIMLoss, VitsDiscriminatorLoss
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.models.base_tts import BaseTTSE2E
from TTS.tts.models.vits import load_audio, wav_to_mel, wav_to_spec
from TTS.tts.utils.emotions import EmotionManager
from TTS.tts.utils.helpers import (
    average_over_durations,
    rand_segments,
    compute_attn_prior,
    segment,
    sequence_mask,
)
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_avg_pitch, plot_pitch, plot_spectrogram
from TTS.utils.audio.numpy_transforms import build_mel_basis, compute_f0
from TTS.utils.audio.numpy_transforms import db_to_amp as db_to_amp_numpy
from TTS.utils.audio.numpy_transforms import mel_to_wav as mel_to_wav_numpy
from TTS.utils.audio.processor import AudioProcessor
from TTS.utils.io import load_fsspec
from TTS.vocoder.layers.losses import MultiScaleSTFTLoss
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results


def id_to_torch(aux_id, cuda=False):
    if aux_id is not None:
        aux_id = np.asarray(aux_id)
        aux_id = torch.from_numpy(aux_id)
    if cuda:
        return aux_id.cuda()
    return aux_id


def embedding_to_torch(d_vector, cuda=False):
    if d_vector is not None:
        d_vector = np.asarray(d_vector)
        d_vector = torch.from_numpy(d_vector).type(torch.FloatTensor)
        d_vector = d_vector.squeeze().unsqueeze(0)
    if cuda:
        return d_vector.cuda()
    return d_vector


def numpy_to_torch(np_array, dtype, cuda=False):
    if np_array is None:
        return None
    tensor = torch.as_tensor(np_array, dtype=dtype)
    if cuda:
        return tensor.cuda()
    return tensor


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


def pad(input_ele: List[torch.Tensor], max_len: int) -> torch.Tensor:
    out_list = torch.jit.annotate(List[torch.Tensor], [])
    for batch in input_ele:
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(batch, (0, max_len - batch.size(0)), "constant", 0.0)
        else:
            one_batch_padded = F.pad(batch, (0, 0, 0, max_len - batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def init_weights(m: nn.Module, mean: float = 0.0, std: float = 0.01):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(mean, std)


def stride_lens(lens: torch.Tensor, stride: int = 2) -> torch.Tensor:
    return torch.ceil(lens / stride).int()


def wav_to_energy(y, n_fft, hop_length, win_length, center=False):
    spec = wav_to_spec(y, n_fft, hop_length, win_length, center=center)
    return torch.norm(spec, dim=1, keepdim=True)


##############################
# DATASET
##############################


def get_attribute_balancer_weights(items: list, attr_name: str, multi_dict: dict = None):
    """Create balancer weight for torch WeightedSampler"""
    attr_names_samples = np.array([item[attr_name] for item in items])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array([len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names])
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    if multi_dict is not None:
        multiplier_samples = np.array([multi_dict.get(item[attr_name], 1.0) for item in items])
        dataset_samples_weight *= multiplier_samples
    return (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )


class ForwardTTSE2eF0Dataset(F0Dataset):
    """Override F0Dataset to avoid the AudioProcessor."""

    def __init__(
        self,
        ap,
        # audio_config: "AudioConfig",
        samples: Union[List[List], List[Dict]],
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        normalize_f0=True,
    ):
        super().__init__(
            ap=ap,
            samples=samples,
            # audio_config=audio_config,
            verbose=verbose,
            cache_path=cache_path,
            precompute_num_workers=precompute_num_workers,
            normalize_f0=normalize_f0,
        )

    @staticmethod
    def _compute_and_save_pitch(ap, wav_file, pitch_file=None):
        wav, _ = load_audio(wav_file)
        f0 = compute_f0(
            x=wav.numpy()[0],
            sample_rate=ap.sample_rate,# audio_config.sample_rate,
            hop_length=ap.hop_length,# audio_config.hop_length,
            pitch_fmax=ap.pitch_fmax,# audio_config.pitch_fmax,
            pitch_fmin=ap.pitch_fmin,
            win_length=ap.win_length
        )
        # skip the last F0 value to align with the spectrogram
        if wav.shape[1] % ap.hop_length != 0:
            f0 = f0[:-1]
        if pitch_file:
            np.save(pitch_file, f0)
        return f0

    # def compute_or_load(self, wav_file):
    #     """
    #     compute pitch and return a numpy array of pitch values
    #     """
    #     pitch_file = self.create_pitch_file_path(wav_file, self.cache_path)
    #     if not os.path.exists(pitch_file):
    #         pitch = self._compute_and_save_pitch(
    #             ap=self.ap, wav_file=wav_file, pitch_file=pitch_file
    #         )
    #     else:
    #         pitch = np.load(pitch_file)
    #     return pitch.astype(np.float32)


class ForwardTTSE2eDataset(TTSDataset):
    def __init__(self, *args, **kwargs):
        # don't init the default F0Dataset in TTSDataset
        compute_f0 = kwargs.pop("compute_f0", False)
        kwargs["compute_f0"] = False
        self.attn_prior_cache_path = kwargs.pop("attn_prior_cache_path")

        super().__init__(*args, **kwargs)

        self.compute_f0 = compute_f0
        self.pad_id = self.tokenizer.characters.pad_id
       #  self.audio_config = kwargs["audio_config"]
        print(self.compute_f0)

        if self.compute_f0:
            self.ap = kwargs['ap']
            self.f0_dataset = ForwardTTSE2eF0Dataset(
                ap=self.ap,
                # audio_config=self.audio_config,
                samples=self.samples,
                cache_path=kwargs["f0_cache_path"],
                precompute_num_workers=kwargs["precompute_num_workers"],
            )

        if self.attn_prior_cache_path is not None:
            os.makedirs(self.attn_prior_cache_path, exist_ok=True)

    def __getitem__(self, idx):
        item = self.samples[idx]

        # prevent unexpected matches by keeping all the folders in the file name
        rel_wav_path = Path(item["audio_file"]).relative_to(item["root_path"]).with_suffix("")
        rel_wav_path = str(rel_wav_path).replace("/", "_")

        raw_text = item["text"]
        wav, _ = load_audio(item["audio_file"])
        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"])

        f0 = None
        if self.compute_f0:
            f0 = self.get_f0(idx)["f0"]

        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)

        # compute attn prior
        attn_prior = None
        if self.attn_prior_cache_path is not None:
            attn_prior = self.load_or_compute_attn_prior(token_ids, wav, rel_wav_path)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "pitch": f0,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "attn_prior": attn_prior,
        }

    def load_or_compute_attn_prior(self, token_ids, wav, rel_wav_path):
        """Load or compute and save the attention prior."""
        attn_prior_file = os.path.join(self.attn_prior_cache_path, f"{rel_wav_path}.npy")
        if os.path.exists(attn_prior_file):
            return np.load(attn_prior_file)
        else:
            token_len = len(token_ids)
            mel_len = wav.shape[1] // self.audio_config.hop_length
            attn_prior = compute_attn_prior(token_len, mel_len)
            np.save(attn_prior_file, attn_prior)
            return attn_prior

    @property
    def lengths(self):
        lens = []
        for item in self.samples:
            _, wav_file, *_ = _parse_sample(item)
            audio_len = os.path.getsize(wav_file) / 16 * 8  # assuming 16bit audio
            lens.append(audio_len)
        return lens

    def collate_fn(self, batch):
        """
        Return Shapes:
            - tokens: :math:`[B, T]`
            - token_lens :math:`[B]`
            - token_rel_lens :math:`[B]`
            - pitch :math:`[B, T]`
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
            - attn_prior: :math:`[[T_token, T_mel]]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        pitch_padded = None
        if self.compute_f0:
            pitch_lens = [p.shape[0] for p in batch["pitch"]]
            pitch_lens = torch.LongTensor(pitch_lens)
            pitch_lens_max = torch.max(pitch_lens)
            pitch_padded = torch.FloatTensor(B, 1, pitch_lens_max)
            pitch_padded = pitch_padded.zero_() + self.pad_id

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)

        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id

        for i in range(B):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

            if self.compute_f0:
                pitch = batch["pitch"][i]
                pitch_padded[i, 0, : len(pitch)] = torch.FloatTensor(pitch)

        return {
            "text_input": token_padded,
            "text_lengths": token_lens,
            "text_rel_lens": token_rel_lens,
            "pitch": pitch_padded,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "attn_priors": batch["attn_prior"] if batch["attn_prior"][0] is not None else None,
        }


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
    lrelu_slope: float


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
    lrelu_slope:float


@dataclass
class VocoderConfig(Coqpit):
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    use_spectral_norm_discriminator: bool = False
    upsampling_rates_discriminator: List[int] = field(default_factory=lambda: [4, 4, 4, 4])
    periods_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    pretrained_model_path: Optional[str] = None


@dataclass
class AcousticModelConfig(Coqpit):
    encoder: ConformerConfig = ConformerConfig(
        n_layers=6,
        n_heads=8,
        n_hidden=512,
        p_dropout=0.1,
        kernel_size_conv_mod=7,
        kernel_size_depthwise=7,
        lrelu_slope=0.3
    )
    decoder: ConformerConfig = ConformerConfig(
        n_layers=6,
        n_heads=8,
        n_hidden=512,
        p_dropout=0.1,
        kernel_size_conv_mod=11,
        kernel_size_depthwise=11,
        lrelu_slope=0.3
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
        n_hidden=512, kernel_size=5, p_dropout=0.5, n_bins=256, emb_kernel_size=3, lrelu_slope=0.3
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
    pitch_fmin: float = 1.0


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



class DelightfulTTSE2e(BaseTTSE2E):
    """
    Model training::
        text --> ForwardTTS() --> spec_hat --> rand_seg_select()--> GANVocoder() --> waveform_seg
        spec --------^

    Examples:
        >>> from TTS.tts.models.forward_tts_e2e import ForwardTTSE2e, ForwardTTSE2eConfig
        >>> config = ForwardTTSE2eConfig()
        >>> model = ForwardTTSE2e(config)
    """

    # pylint: disable=dangerous-default-value
    def __init__(
        self,
        config: Coqpit,
        ap,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        emotion_manager: EmotionManager = None,
    ):
        super().__init__(config=config, ap=ap, tokenizer=tokenizer, speaker_manager=speaker_manager)
        self.emotion_manager = emotion_manager
        self.ap = ap

        self._set_model_args(config)
        self.init_multispeaker(config)
        self.binary_loss_weight = None

        self.args.acoustic_config.out_channels = self.config.audio.num_mels
        self.args.acoustic_config.num_mels = self.config.audio.num_mels
        self.acoustic_model = AcousticModel(
            args=self.args, tokenizer=tokenizer, speaker_manager=speaker_manager, emotion_manager=emotion_manager
        )

        self.waveform_decoder = HifiganGenerator(
            self.config.audio.num_mels,
            1,
            self.config.vocoder.resblock_type_decoder,
            self.config.vocoder.resblock_dilation_sizes_decoder,
            self.config.vocoder.resblock_kernel_sizes_decoder,
            self.config.vocoder.upsample_kernel_sizes_decoder,
            self.config.vocoder.upsample_initial_channel_decoder,
            self.config.vocoder.upsample_rates_decoder,
            inference_padding=0,
            # cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        # use Vits Discriminator for limiting VRAM use
        if self.config.init_discriminator:
            self.disc = VitsDiscriminator(
                use_spectral_norm=self.config.vocoder.use_spectral_norm_discriminator,
                periods=self.config.vocoder.periods_discriminator,
                upsampling_rates=self.config.vocoder.upsampling_rates_discriminator,
            )

    def init_for_training(self):
        ...

    # def init_vocoder_from_pretrained(self, model_path:str):
    #     pretrained_dict = load_fsspec(model_path)
    #     if "model" in pretrained_dict:
    #         pretrained_dict = pretrained_dict["model"]

    #     print(" > Initializing vocoder from pretrained model: {}".format(model_path))
    #     sub_pretrained_dict = {}
    #     for name, param in pretrained_dict:
    #         if name.startswith("waveform_decoder."):
    #             new_name = name.replace("waveform_decoder.", "")
    #             sub_pretrained_dict[new_name] = param
    #     self.waveform_decoder.load_state_dict(sub_pretrained_dict, strict=False)

    #     if self.config.init_discriminator:
    #         print(" > Initializing vocoder discriminator from pretrained model: {}".format(model_path))
    #         sub_pretrained_dict = {}
    #         for name, param in pretrained_dict:
    #             if name.startswith("disc."):
    #                 new_name = name.replace("disc.", "")
    #                 sub_pretrained_dict[new_name] = param
    #         self.disc.load_state_dict(sub_pretrained_dict, strict=False)

    # def on_init_end(self, trainer):
    #     self.init_vocoder_from_pretrained(self.config.vocoder.pretrained_model_path)

    @property
    def energy_scaler(self):
        return self.acoustic_model.energy_scaler

    @property
    def length_scale(self):
        return self.acoustic_model.length_scale

    @length_scale.setter
    def length_scale(self, value):
        self.acoustic_model.length_scale = value

    @property
    def pitch_mean(self):
        return self.acoustic_model.pitch_mean

    @pitch_mean.setter
    def pitch_mean(self, value):
        self.acoustic_model.pitch_mean = value

    @property
    def pitch_std(self):
        return self.acoustic_model.pitch_std

    @pitch_mean.setter
    def pitch_std(self, value):
        self.acoustic_model.pitch_std = value

    @property
    def mel_basis(self):
        return build_mel_basis(
            sample_rate=self.config.audio.sample_rate,
            fft_size=self.config.audio.fft_size,
            num_mels=self.config.audio.num_mels,
            mel_fmax=self.config.audio.mel_fmax,
            mel_fmin=self.config.audio.mel_fmin,
        )

    def init_for_training(self) -> None:
        self.train_disc = self.config.steps_to_start_discriminator <= 0
        self.update_energy_scaler = True

    def init_multispeaker(self, config: Coqpit):
        """Init for multi-speaker training.

        Args:
            config (Coqpit): Model configuration.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers
            self.args.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.args.embedded_speaker_dim = self.args.speaker_embedding_channels

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim
        self.args.embedded_speaker_dim = self.args.d_vector_dim

    def _freeze_layers(self):
        # TODO: freeze layers
        ...

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        spec_lengths: torch.LongTensor,
        spec: torch.FloatTensor,
        waveform: torch.FloatTensor,
        pitch: torch.FloatTensor = None,
        energy: torch.FloatTensor = None,
        attn_priors: torch.FloatTensor = None,
        d_vectors: torch.FloatTensor = None,
        emo_vectors: torch.FloatTensor = None,
        speaker_idx: torch.LongTensor = None,
    ) -> Dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            x_lengths (torch.LongTensor): Input sequence lengths.
            spec_lengths (torch.LongTensor): Spectrogram sequnce lengths. Defaults to None.
            spec (torch.FloatTensor): Spectrogram frames. Only used when the alignment network is on. Defaults to None.
            waveform (torch.FloatTensor): Waveform. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Only used when the pitch predictor is on. Defaults to None.
            energy (torch.FloatTensor): Spectral energy values for each spectrogram frame. Only used when the energy predictor is on. Defaults to None.
            attn_priors (torch.FloatTentrasor): Attention priors for the aligner network. Defaults to None.
            aux_input (Dict): Auxiliary model inputs for multi-speaker training. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - spec_lengths: :math:`[B]`
            - spec: :math:`[B, T_max2, C_spec]`
            - waveform: :math:`[B, 1, T_max2 * hop_length]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T_max2]`
            - energy: :math:`[B, 1, T_max2]`
        """
        encoder_outputs = self.acoustic_model(
            tokens=x,
            src_lens=x_lengths,
            mel_lens=spec_lengths,
            mels=spec,
            pitches=pitch,
            energies=energy,
            attn_priors=attn_priors,
            d_vectors=d_vectors,
            emo_vectors=emo_vectors,
            speaker_idx=speaker_idx,
        )

        # use mel-spec from the decoder
        vocoder_input = encoder_outputs["model_outputs"]  # [B, T_max2, C_mel]

        vocoder_input_slices, slice_ids = rand_segments(
            x=vocoder_input.transpose(1, 2),
            x_lengths=spec_lengths,
            segment_size=self.args.spec_segment_size,
            let_short_samples=True,
            pad_short=True,
        )

        # TODO: not sure if we need to pass spk_emb to the vocoder
        vocoder_output = self.waveform_decoder(
            x=vocoder_input_slices.detach(),
        )
        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.args.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )
        model_outputs = {**encoder_outputs}
        model_outputs["acoustic_model_outputs"] = encoder_outputs["model_outputs"]
        model_outputs["model_outputs"] = vocoder_output
        model_outputs["waveform_seg"] = wav_seg
        model_outputs["slice_ids"] = slice_ids
        return model_outputs

    @torch.no_grad()
    def inference(self, x, d_vectors=None, emotion_vectors=None, speaker_idx=None, pitch_transform=None, energy_transform=None):
        encoder_outputs = self.acoustic_model.inference(
            tokens=x,
            d_vectors=d_vectors,
            emo_vectors=emotion_vectors,
            speaker_idx=speaker_idx,
            pitch_transform=pitch_transform,
            energy_transform=energy_transform,
            p_control=None,
            d_control=None,
        )
        vocoder_input = encoder_outputs["model_outputs"].transpose(1, 2)  # [B, T_max2, C_mel] -> [B, C_mel, T_max2]
        vocoder_output = self.waveform_decoder(x=vocoder_input)
        model_outputs = {**encoder_outputs}
        model_outputs["model_outputs"] = vocoder_output
        return model_outputs

    @torch.no_grad()
    def inference_spec_decoder(self, x, d_vectors=None, emotion_vectors=None, speaker_idx=None):
        encoder_outputs = self.acoustic_model.inference(
            tokens=x,
            speaker_idx=speaker_idx,
            d_vectors=d_vectors,
            emo_vectors=emotion_vectors,
        )
        model_outputs = {**encoder_outputs}
        return model_outputs

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        if optimizer_idx == 0:
            tokens = batch["text_input"]
            token_lenghts = batch["text_lengths"]
            mel = batch["mel_input"]
            mel_lens = batch["mel_lengths"]
            waveform = batch["waveform"]  # [B, T, C] -> [B, C, T]
            pitch = batch["pitch"]
            d_vectors = batch["d_vectors"]
            emo_vectors = batch["emo_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]
            attn_priors = batch["attn_priors"]
            energy = batch["energy"]

            # generator pass
            outputs = self.forward(
                x=tokens,
                x_lengths=token_lenghts,
                spec_lengths=mel_lens,
                spec=mel,
                waveform=waveform,
                pitch=pitch,
                energy=energy,
                attn_priors=attn_priors,
                d_vectors=d_vectors,
                emo_vectors=emo_vectors,
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            if self.train_disc:
                # compute scores and features
                scores_d_fake, _, scores_d_real, _ = self.disc(
                    outputs["model_outputs"].detach(), outputs["waveform_seg"]
                )

                # compute loss
                with autocast(enabled=False):  # use float32 for the criterion
                    loss_dict = criterion[optimizer_idx](
                        scores_disc_fake=scores_d_fake,
                        scores_disc_real=scores_d_real,
                    )
                return outputs, loss_dict
            return None, None

        if optimizer_idx == 1:
            mel = batch["mel_input"]
            # compute melspec segment
            with autocast(enabled=False):
                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], self.args.spec_segment_size, pad_short=True
                )

                mel_slice_hat = wav_to_mel(
                    y=self.model_outputs_cache["model_outputs"].float(),
                    n_fft=self.config.audio.fft_size,
                    sample_rate=self.config.audio.sample_rate,
                    num_mels=self.config.audio.num_mels,
                    hop_length=self.config.audio.hop_length,
                    win_length=self.config.audio.win_length,
                    fmin=self.config.audio.mel_fmin,
                    fmax=self.config.audio.mel_fmax,
                    center=False,
                )

                scores_d_fake = None
                feats_d_fake = None
                feats_d_real = None

            if self.train_disc:
                # compute discriminator scores and features
                scores_d_fake, feats_d_fake, _, feats_d_real = self.disc(
                    self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
                )

            # compute losses
            with autocast(enabled=True):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_output=self.model_outputs_cache["acoustic_model_outputs"].transpose(1, 2),
                    mel_target=batch["mel_input"],
                    mel_lens=batch["mel_lengths"],
                    dur_output=self.model_outputs_cache["dr_log_pred"],
                    dur_target=self.model_outputs_cache["dr_log_target"].detach(),
                    pitch_output=self.model_outputs_cache["pitch_pred"],
                    pitch_target=self.model_outputs_cache["pitch_target"],
                    energy_output=self.model_outputs_cache["energy_pred"],
                    energy_target=self.model_outputs_cache["energy_target"],
                    src_lens=batch["text_lengths"],
                    waveform=self.model_outputs_cache["waveform_seg"],
                    waveform_hat=self.model_outputs_cache["model_outputs"],
                    p_prosody_ref=self.model_outputs_cache["p_prosody_ref"],
                    p_prosody_pred=self.model_outputs_cache["p_prosody_pred"],
                    u_prosody_ref=self.model_outputs_cache["u_prosody_ref"],
                    u_prosody_pred=self.model_outputs_cache["u_prosody_pred"],
                    aligner_logprob=self.model_outputs_cache["aligner_logprob"],
                    aligner_hard=self.model_outputs_cache["aligner_mas"],
                    aligner_soft=self.model_outputs_cache["aligner_soft"],
                    binary_loss_weight=self.binary_loss_weight,
                    feats_fake=feats_d_fake,
                    feats_real=feats_d_real,
                    scores_fake=scores_d_fake,
                    spec_slice=mel_slice,
                    spec_slice_hat=mel_slice_hat,
                    skip_disc=not self.train_disc,
                )

                # compute duration error for logging
                # durations_pred = self.model_outputs_cache["dr_pred"]
                # durations_target = self.model_outputs_cache["dr_target"]
                # duration_error = torch.abs(durations_target - durations_pred).sum() / batch["text_lengths"].sum()
                # loss_dict["duration_error"] = duration_error

                loss_dict["avg_text_length"] = batch["text_lengths"].float().mean()
                loss_dict["avg_mel_length"] = batch["mel_lengths"].float().mean()
                loss_dict["avg_text_batch_occupancy"] = (
                    batch["text_lengths"].float() / batch["text_lengths"].float().max()
                ).mean()
                loss_dict["avg_mel_batch_occupancy"] = (
                    batch["mel_lengths"].float() / batch["mel_lengths"].float().max()
                ).mean()

            return self.model_outputs_cache, loss_dict
        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def _log(self, batch, outputs, name_prefix="train"):
        figures, audios = {}, {}

        # encoder outputs
        model_outputs = outputs[1]["acoustic_model_outputs"]
        alignments = outputs[1]["alignments"]
        mel_input = batch["mel_input"]

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, None, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec.T, None, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        # plot pitch figures
        pitch_avg = abs(outputs[1]["pitch_target"][0, 0].data.cpu().numpy())
        pitch_avg_hat = abs(outputs[1]["pitch_pred"][0, 0].data.cpu().numpy())
        chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
        pitch_figures = {
            "pitch_ground_truth": plot_avg_pitch(pitch_avg, chars, output_fig=False),
            "pitch_avg_predicted": plot_avg_pitch(pitch_avg_hat, chars, output_fig=False),
        }
        figures.update(pitch_figures)

        # plot energy figures
        energy_avg = abs(outputs[1]["energy_target"][0, 0].data.cpu().numpy())
        energy_avg_hat = abs(outputs[1]["energy_pred"][0, 0].data.cpu().numpy())
        chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
        energy_figures = {
            "energy_ground_truth": plot_avg_pitch(energy_avg, chars, output_fig=False),
            "energy_avg_predicted": plot_avg_pitch(energy_avg_hat, chars, output_fig=False),
        }
        figures.update(energy_figures)

        # plot the attention mask computed from the predicted durations
        alignments_hat = outputs[1]["alignments_dp"][0].data.cpu().numpy()
        figures["alignment_hat"] = plot_alignment(alignments_hat.T, output_fig=False)

        # Sample audio
        encoder_audio = mel_to_wav_numpy(
            mel=db_to_amp_numpy(x=pred_spec.T, gain=1, base=None), mel_basis=self.mel_basis, **self.config.audio
        )
        audios[f"{name_prefix}/encoder_audio"] = encoder_audio

        # vocoder outputs
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]

        vocoder_figures = plot_results(y_hat=y_hat, y=y, audio_config=self.config.audio, name_prefix=name_prefix)
        figures.update(vocoder_figures)

        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios[f"{name_prefix}/vocoder_audio"] = sample_voice
        return figures, audios

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use, unused-argument
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previous training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        figures, audios = self._log(batch=batch, outputs=outputs, name_prefix="vocoder/")
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.config.audio.sample_rate)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._log(batch=batch, outputs=outputs, name_prefix="vocoder/")
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.config.audio.sample_rate)

    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name = None, None, None, None

        if isinstance(sentence_info, list):
            if len(sentence_info) == 1:
                text = sentence_info[0]
            elif len(sentence_info) == 2:
                text, speaker_name = sentence_info
            elif len(sentence_info) == 3:
                text, speaker_name, style_wav = sentence_info
            elif len(sentence_info) == 4:
                text, speaker_name, style_wav, language_name = sentence_info
            elif len(sentence_info) == 5:
                text, speaker_name, style_wav, language_name, emotion = sentence_info
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embeddings()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.ids[speaker_name]

        # get emotion id/vector
        emotion_vector = None
        if hasattr(self, "speaker_manager"):
            if config.use_emotion_vector_file:
                if emotion is None:
                    emotion_vector = self.emotion_manager.get_random_embeddings()
                else:
                    emotion_vector = self.emotion_manager.get_mean_embedding(emotion, num_samples=None, randomize=False)

        # get language id
        # if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
        #     language_id = self.language_manager.ids[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": None,
            "language_name": None,
            "emotion_vector": emotion_vector,
        }

    def plot_outputs(self, text, wav, alignment, outputs):
        figures = {}
        pitch_avg_pred = outputs["pitch"].cpu()
        energy_avg_pred = outputs["energy"].cpu()
        spec = wav_to_mel(
            y=torch.from_numpy(wav[None, :]),
            n_fft=self.config.audio.fft_size,
            sample_rate=self.config.audio.sample_rate,
            num_mels=self.config.audio.num_mels,
            hop_length=self.config.audio.hop_length,
            win_length=self.config.audio.win_length,
            fmin=self.config.audio.mel_fmin,
            fmax=self.config.audio.mel_fmax,
            center=False,
        )[0].transpose(0, 1)
        pitch = compute_f0(
            x=wav[0],
            sample_rate=self.config.audio.sample_rate,
            hop_length=self.config.audio.hop_length,
            pitch_fmax=self.config.audio.pitch_fmax,
        )
        input_text = self.tokenizer.ids_to_text(self.tokenizer.text_to_ids(text, language="en"))
        input_text = input_text.replace("<BLNK>", "_")
        durations = outputs["durations"]
        pitch_avg = average_over_durations(
            torch.from_numpy(pitch)[None, None, :], durations.cpu()
        )  # [1, 1, n_frames]
        pitch_avg_pred_denorm = (pitch_avg_pred * self.pitch_std) + self.pitch_mean
        figures["alignment"] = plot_alignment(alignment.transpose(1, 2), output_fig=False)
        figures["spectrogram"] = plot_spectrogram(spec)
        figures["pitch_from_wav"] = plot_pitch(pitch, spec)
        figures["pitch_avg_from_wav"] = plot_avg_pitch(pitch_avg.squeeze(), input_text)
        figures["pitch_avg_pred"] = plot_avg_pitch(pitch_avg_pred_denorm.squeeze(), input_text)
        figures["energy_avg_pred"] = plot_avg_pitch(energy_avg_pred.squeeze(), input_text)
        return figures

    def synthesize(self, text: str, speaker_id, language_id, emotion_id, d_vector, emotion_vector, ref_waveform=None, pitch_transform=None):
        # TODO: add language_id
        is_cuda = next(self.parameters()).is_cuda

        # convert text to sequence of token IDs
        text_inputs = np.asarray(
            self.tokenizer.text_to_ids(text, language=language_id),
            dtype=np.int32,
        )

        # pass tensors to backend
        if speaker_id is not None:
            speaker_id = id_to_torch(speaker_id, cuda=is_cuda)

        if d_vector is not None:
            d_vector = embedding_to_torch(d_vector, cuda=is_cuda)

        if emotion_vector is not None:
            emotion_vector = embedding_to_torch(emotion_vector, cuda=is_cuda)

        # if language_id is not None:
        #     language_id = id_to_torch(language_id, cuda=is_cuda)

        text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=is_cuda)
        text_inputs = text_inputs.unsqueeze(0)

        # synthesize voice
        outputs = self.inference(
            text_inputs,
            d_vectors=d_vector,
            emotion_vectors=emotion_vector,
            speaker_idx=speaker_id,
            pitch_transform=pitch_transform,
            # energy_transform=energy_transform
        )

        # collect outputs
        wav = outputs["model_outputs"][0].data.cpu().numpy()
        alignments = outputs["alignments"]
        return_dict = {
            "wav": wav,
            "alignments": alignments,
            "text_inputs": text_inputs,
            "outputs": outputs,
        }
        return return_dict

    def synthesize_with_gl(self, text: str, speaker_id, language_id, d_vector, emotion_vector):

        # TODO: add language_id
        is_cuda = next(self.parameters()).is_cuda

        # convert text to sequence of token IDs
        text_inputs = np.asarray(
            self.tokenizer.text_to_ids(text, language=language_id),
            dtype=np.int32,
        )
        # pass tensors to backend
        if speaker_id is not None:
            speaker_id = id_to_torch(speaker_id, cuda=is_cuda)

        if d_vector is not None:
            d_vector = embedding_to_torch(d_vector, cuda=is_cuda)

        if emotion_vector is not None:
            emotion_vector = embedding_to_torch(emotion_vector, cuda=is_cuda)

        # if language_id is not None:
        #     language_id = id_to_torch(language_id, cuda=is_cuda)

        text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=is_cuda)
        text_inputs = text_inputs.unsqueeze(0)

        # synthesize voice
        outputs = self.inference_spec_decoder(
            x=text_inputs,
            d_vectors=d_vector,
            emotion_vectors=emotion_vector,
            speaker_idx=speaker_id,
        )

        # collect outputs
        S = outputs["model_outputs"].cpu().numpy()[0].T
        S = db_to_amp_numpy(x=S, gain=1, base=None)
        wav = mel_to_wav_numpy(mel=S, mel_basis=self.mel_basis, **self.config.audio)
        alignments = outputs["alignments"]
        return_dict = {
            "wav": wav[None, :],
            "alignments": alignments,
            "text_inputs": text_inputs,
            "outputs": outputs,
        }
        return return_dict

    @torch.no_grad()
    def test_run(self, assets) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        for idx, s_info in enumerate(test_sentences):
            aux_inputs = self.get_aux_input_from_test_sentences(s_info)

            print(f"this is auxs: {aux_inputs}")
            outputs = self.synthesize(
                aux_inputs["text"],
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                language_id=aux_inputs["language_id"],
                emotion_vector=aux_inputs["emotion_vector"],
                emotion_id=None
            )
            outputs_gl = self.synthesize_with_gl(
                aux_inputs["text"],
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                language_id=aux_inputs["language_id"],
                emotion_vector=aux_inputs["emotion_vector"],
            )
            # speaker_name = self.speaker_manager.speaker_names[aux_inputs["speaker_id"]]
            test_audios["{}-audio".format(idx)] = outputs["wav"].T
            test_audios["{}-audio_encoder".format(idx)] = outputs_gl["wav"].T
            test_figures["{}-alignment".format(idx)] = plot_alignment(outputs["alignments"], output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.config.audio.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs and d_vector for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None
        emotion_vectors = None

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.speaker_names and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.speaker_names[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            batch["speaker_ids"] = speaker_ids

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_files"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if (
            self.language_manager is not None
            and self.language_manager.language_id_mapping
            and self.args.use_language_embedding
        ):
            language_ids = [self.language_manager.language_id_mapping[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        # get emotions
        if self.emotion_manager is not None and self.emotion_manager.embeddings and self.args.use_emotion_vector_file:
            emotion_mapping = self.emotion_manager.embeddings
            emotion_vectors = [emotion_mapping[w]["embedding"] for w in batch["audio_files"]]
            emotion_vectors = torch.FloatTensor(emotion_vectors)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["emo_vectors"] = emotion_vectors
        batch["speaker_ids"] = speaker_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""

        ac = self.config.audio

        # compute spectrograms
        batch["mel_input"] = wav_to_mel(
            batch["waveform"],
            hop_length=ac.hop_length,
            win_length=ac.win_length,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
            center=False,
        )

        # TODO: Align pitch properly
        # assert (
        #     batch["pitch"].shape[2] == batch["mel_input"].shape[2]
        # ), f"{batch['pitch'].shape[2]}, {batch['mel_input'].shape[2]}"
        batch["pitch"] = batch["pitch"][:, :, : batch["mel_input"].shape[2]] if batch["pitch"] is not None else None
        batch["mel_lengths"] = (batch["mel_input"].shape[2] * batch["waveform_rel_lens"]).int()

        # zero the padding frames
        batch["mel_input"] = batch["mel_input"] * sequence_mask(batch["mel_lengths"]).unsqueeze(1)

        # format attn priors as we now the max mel length
        # TODO: fix 1 diff b/w mel_lengths and attn_priors

        if self.config.use_attn_priors:
            attn_priors_np = batch["attn_priors"]

            batch["attn_priors"] = torch.zeros(
                batch["mel_input"].shape[0],
                batch["mel_lengths"].max(),
                batch["text_lengths"].max(),
                device=batch["mel_input"].device,
            )
            print(f'this is the attention priors np: {attn_priors_np}')
            print(f"this is the attention priors: {batch['attn_priors']}")

            for i in range(batch["mel_input"].shape[0]):
                batch["attn_priors"][i, : attn_priors_np[i].shape[0], : attn_priors_np[i].shape[1]] = torch.from_numpy(
                    attn_priors_np[i]
                )

        batch["energy"] = None
        batch["energy"] = wav_to_energy(  # [B, 1, T_max2]
            batch["waveform"],
            hop_length=ac.hop_length,
            win_length=ac.win_length,
            n_fft=ac.fft_size,
            center=False,
        )
        batch["energy"] = self.energy_scaler(batch["energy"])
        return batch

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1):
        weights = None
        data_items = dataset.samples
        if getattr(config, "use_weighted_sampler", False):
            for attr_name, alpha in config.weighted_sampler_attrs.items():
                print(f" > Using weighted sampler for attribute '{attr_name}' with alpha '{alpha}'")
                multi_dict = config.weighted_sampler_multipliers.get(attr_name, None)
                print(multi_dict)
                weights, attr_names, attr_weights = get_attribute_balancer_weights(
                    attr_name=attr_name, items=data_items, multi_dict=multi_dict
                )
                weights = weights * alpha
                print(f" > Attribute weights for '{attr_names}' \n | > {attr_weights}")

        if weights is not None:
            sampler = WeightedRandomSampler(weights, len(weights))
        else:
            sampler = None
        # sampler for DDP
        if sampler is None:
            sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            sampler = DistributedSamplerWrapper(sampler) if num_gpus > 1 else sampler
        return sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: Dict,
        is_eval: bool,
        samples: Union[List[Dict], List[List]],
        verbose: bool,
        num_gpus: int,
        rank: int = None,
    ) -> "DataLoader":
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = ForwardTTSE2eDataset(
                samples=samples,
                ap=self.ap,
                # audio_config=self.config.audio,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                compute_f0=config.compute_f0,
                f0_cache_path=config.f0_cache_path,
                attn_prior_cache_path=config.attn_prior_cache_path if config.use_attn_priors else None,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences ascendingly by length
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)

            loader = DataLoader(
                dataset,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                shuffle=False,  # shuffle is done in the dataset.
                drop_last=False,  # setting this False might cause issues in AMP training.
                sampler=sampler,
                collate_fn=dataset.collate_fn,
                num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                pin_memory=True,
            )

            # get pitch mean and std
            self.pitch_mean = dataset.f0_dataset.mean
            self.pitch_std = dataset.f0_dataset.std
        return loader

    def get_criterion(self):
        return [VitsDiscriminatorLoss(self.config), SomethingTTSLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        optimizer_disc = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc
        )
        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer_gen = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer_disc, optimizer_gen]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_disc, self.config.lr_gen]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler_D = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def on_train_step_start(self, trainer):
        """Schedule binary loss weight."""
        self.binary_loss_weight = min(trainer.epochs_done / self.config.binary_loss_warmup_epochs, 1.0) * 1.0

    def on_epoch_end(self, trainer):
        # stop updating mean and var
        # TODO: do the same for F0
        self.energy_scaler.eval()

    @staticmethod
    def init_from_config(config: "SomethingTTSConfig", samples: Union[List[List], List[Dict]] = None, verbose=False):
        """Initiate model from config

        Args:
            config (ForwardTTSE2eConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """

        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config.model_args, samples)
        emotion_manager = EmotionManager.init_from_config(config.model_args)
        # language_manager = LanguageManager.init_from_config(config)
        return DelightfulTTSE2e(
            config=new_config, tokenizer=tokenizer, speaker_manager=speaker_manager, emotion_manager=emotion_manager
        )

    def load_checkpoint(self, config, checkpoint_path, eval=False):
        """Load model from a checkpoint created by the 👟"""
        # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_state_dict(self):
        """Custom state dict of the model with all the necessary components for inference."""
        save_state = {"config": self.config.to_dict(), "args": self.args.to_dict(), "model": self.state_dict}

        if hasattr(self, "emb_g"):
            save_state["speaker_ids"] = self.speaker_manager.speaker_names

        if self.args.use_d_vector_file:
            # TODO: implement saving of d_vectors
            ...
        return save_state

    def save(self, config, checkpoint_path):
        """Save model to a file."""
        save_state = self.get_state_dict(config, checkpoint_path)
        save_state["pitch_mean"] = self.pitch_mean
        save_state["pitch_std"] = self.pitch_std
        torch.save(save_state, checkpoint_path)

    def on_train_step_start(self, trainer) -> None:
        """Enable the discriminator training based on `steps_to_start_discriminator`

        Args:
            trainer (Trainer): Trainer object.
        """
        self.train_disc = trainer.total_steps_done >= self.config.steps_to_start_discriminator


class SomethingTTSLoss(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()
        self.ssim_loss = SSIMLoss()
        self.forward_sum_loss = ForwardSumLoss()
        self.multi_scale_stft_loss = MultiScaleSTFTLoss(**config.multi_scale_stft_loss_params)

        self.ssim_loss_alpha = config.ssim_loss_alpha
        self.mel_loss_alpha = config.mel_loss_alpha
        self.aligner_loss_alpha = config.aligner_loss_alpha
        self.pitch_loss_alpha = config.pitch_loss_alpha
        self.energy_loss_alpha = config.energy_loss_alpha
        self.u_prosody_loss_alpha = config.u_prosody_loss_alpha
        self.p_prosody_loss_alpha = config.p_prosody_loss_alpha
        self.dur_loss_alpha = config.dur_loss_alpha
        self.char_dur_loss_alpha = config.char_dur_loss_alpha
        self.binary_alignment_loss_alpha = config.binary_align_loss_alpha

        self.vocoder_mel_loss_alpha = config.vocoder_mel_loss_alpha
        self.feat_loss_alpha = config.feat_loss_alpha
        self.gen_loss_alpha = config.gen_loss_alpha
        self.multi_scale_stft_loss_alpha = config.multi_scale_stft_loss_alpha

    @staticmethod
    def _binary_alignment_loss(alignment_hard, alignment_soft):
        """Binary loss that forces soft alignments to match the hard alignments as
        explained in `https://arxiv.org/pdf/2108.10447.pdf`.
        """
        log_sum = torch.log(torch.clamp(alignment_soft[alignment_hard == 1], min=1e-12)).sum()
        return -log_sum / alignment_hard.sum()

    @staticmethod
    def feature_loss(feats_real, feats_generated):
        loss = 0
        for dr, dg in zip(feats_real, feats_generated):
            for rl, gl in zip(dr, dg):
                rl = rl.float().detach()
                gl = gl.float()
                loss += torch.mean(torch.abs(rl - gl))
        return loss * 2

    @staticmethod
    def generator_loss(scores_fake):
        loss = 0
        gen_losses = []
        for dg in scores_fake:
            dg = dg.float()
            l = torch.mean((1 - dg) ** 2)
            gen_losses.append(l)
            loss += l

        return loss, gen_losses

    def forward(
        self,
        mel_output,
        mel_target,
        mel_lens,
        dur_output,
        dur_target,
        pitch_output,
        pitch_target,
        energy_output,
        energy_target,
        src_lens,
        waveform,
        waveform_hat,
        p_prosody_ref,
        p_prosody_pred,
        u_prosody_ref,
        u_prosody_pred,
        aligner_logprob,
        aligner_hard,
        aligner_soft,
        binary_loss_weight=None,
        feats_fake=None,
        feats_real=None,
        scores_fake=None,
        spec_slice=None,
        spec_slice_hat=None,
        skip_disc=False,
    ):
        """
        Shapes:
            - mel_output: :math:`(B, C_mel, T_mel)`
            - mel_target: :math:`(B, C_mel, T_mel)`
            - mel_lens: :math:`(B)`
            - dur_output: :math:`(B, T_src)`
            - dur_target: :math:`(B, T_src)`
            - pitch_output: :math:`(B, 1, T_src)`
            - pitch_target: :math:`(B, 1, T_src)`
            - energy_output: :math:`(B, 1, T_src)`
            - energy_target: :math:`(B, 1, T_src)`
            - src_lens: :math:`(B)`
            - waveform: :math:`(B, 1, T_wav)`
            - waveform_hat: :math:`(B, 1, T_wav)`
            - p_prosody_ref: :math:`(B, T_src, 4)`
            - p_prosody_pred: :math:`(B, T_src, 4)`
            - u_prosody_ref: :math:`(B, 1, 256)
            - u_prosody_pred: :math:`(B, 1, 256)
            - aligner_logprob: :math:`(B, 1, T_mel, T_src)`
            - aligner_hard: :math:`(B, T_mel, T_src)`
            - aligner_soft: :math:`(B, T_mel, T_src)`
            - spec_slice: :math:`(B, C_mel, T_mel)`
            - spec_slice_hat: :math:`(B, C_mel, T_mel)`
        """
        loss_dict = {}
        src_mask = sequence_mask(src_lens).to(mel_output.device)  # (B, T_src)
        mel_mask = sequence_mask(mel_lens).to(mel_output.device)  # (B, T_mel)

        dur_target.requires_grad = False
        mel_target.requires_grad = False
        pitch_target.requires_grad = False

        ssim_loss = self.ssim_loss(mel_output, mel_target, mel_lens)

        masked_mel_predictions = mel_output.masked_select(mel_mask[:, None])
        mel_targets = mel_target.masked_select(mel_mask[:, None])
        mel_loss = self.mae_loss(masked_mel_predictions, mel_targets)

        p_prosody_ref = p_prosody_ref.detach()
        p_prosody_loss = 0.5 * self.mae_loss(
            p_prosody_ref.masked_select(src_mask.unsqueeze(-1)),
            p_prosody_pred.masked_select(src_mask.unsqueeze(-1)),
        )

        u_prosody_ref = u_prosody_ref.detach()
        u_prosody_loss = 0.5 * self.mae_loss(u_prosody_ref, u_prosody_pred)

        duration_loss = self.mse_loss(dur_output, dur_target)

        pitch_output = pitch_output.masked_select(src_mask[:, None])
        pitch_target = pitch_target.masked_select(src_mask[:, None])
        pitch_loss = self.mse_loss(pitch_output, pitch_target)

        energy_output = energy_output.masked_select(src_mask[:, None])
        energy_target = energy_target.masked_select(src_mask[:, None])
        energy_loss = self.mse_loss(energy_output, energy_target)

        forward_sum_loss = self.forward_sum_loss(aligner_logprob, src_lens, mel_lens)

        total_loss = (
            (mel_loss * self.mel_loss_alpha)
            + (duration_loss * self.dur_loss_alpha)
            + (u_prosody_loss * self.u_prosody_loss_alpha)
            + (p_prosody_loss * self.p_prosody_loss_alpha)
            + (ssim_loss * self.ssim_loss_alpha)
            + (pitch_loss * self.pitch_loss_alpha)
            + (energy_loss * self.energy_loss_alpha)
            + (forward_sum_loss * self.aligner_loss_alpha)
        )

        if self.binary_alignment_loss_alpha > 0 and aligner_hard is not None:
            binary_alignment_loss = self._binary_alignment_loss(aligner_hard, aligner_soft)
            total_loss = total_loss + self.binary_alignment_loss_alpha * binary_alignment_loss * binary_loss_weight
            if binary_loss_weight:
                loss_dict["loss_binary_alignment"] = (
                    self.binary_alignment_loss_alpha * binary_alignment_loss * binary_loss_weight
                )
            else:
                loss_dict["loss_binary_alignment"] = self.binary_alignment_loss_alpha * binary_alignment_loss

        loss_dict["loss_aligner"] = self.aligner_loss_alpha * forward_sum_loss
        loss_dict["loss_mel"] = self.mel_loss_alpha * mel_loss
        loss_dict["loss_duration"] = self.dur_loss_alpha * duration_loss
        loss_dict["loss_u_prosody"] = self.u_prosody_loss_alpha * u_prosody_loss
        loss_dict["loss_p_prosody"] = self.p_prosody_loss_alpha * p_prosody_loss
        loss_dict["loss_ssim"] = self.ssim_loss_alpha * ssim_loss
        loss_dict["loss_pitch"] = self.pitch_loss_alpha * pitch_loss
        loss_dict["loss_energy"] = self.energy_loss_alpha * energy_loss
        loss_dict["loss"] = total_loss

        # vocoder losses
        if not skip_disc:
            loss_feat = self.feature_loss(feats_real=feats_real, feats_generated=feats_fake) * self.feat_loss_alpha
            loss_gen = self.generator_loss(scores_fake=scores_fake)[0] * self.gen_loss_alpha
            loss_dict["vocoder_loss_feat"] = loss_feat
            loss_dict["vocoder_loss_gen"] = loss_gen
            loss_dict["loss"] = loss_dict["loss"] + loss_feat + loss_gen

        loss_mel = torch.nn.functional.l1_loss(spec_slice, spec_slice_hat) * self.vocoder_mel_loss_alpha
        loss_stft_mg, loss_stft_sc = self.multi_scale_stft_loss(y_hat=waveform_hat, y=waveform)
        loss_stft_mg = loss_stft_mg * self.multi_scale_stft_loss_alpha
        loss_stft_sc = loss_stft_sc * self.multi_scale_stft_loss_alpha

        loss_dict["vocoder_loss_mel"] = loss_mel
        loss_dict["vocoder_loss_stft_mg"] = loss_stft_mg
        loss_dict["vocoder_loss_stft_sc"] = loss_stft_sc

        loss_dict["loss"] = loss_dict["loss"] + loss_mel + loss_stft_sc + loss_stft_mg
        return loss_dict