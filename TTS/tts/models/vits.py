import math
import os
import numpy as np
import pyworld as pw
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Callable, Dict, List, Tuple, Union

import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample, F0Dataset
from TTS.tts.layers.generic.classifier import ReversalClassifier
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.glow_tts.transformer import RelativePositionTransformer
from TTS.tts.layers.feed_forward.decoder import Decoder as forwardDecoder
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.layers.vits.prosody_encoder import VitsGST, VitsVAE, ResNetProsodyEncoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.data import prepare_data
from TTS.tts.utils.helpers import average_over_durations
from TTS.tts.utils.emotions import EmotionManager
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results

##############################
# IO / Feature extraction
##############################

# pylint: disable=global-statement
hann_window = {}
mel_basis = {}


@torch.no_grad()
def weights_reset(m: nn.Module):
    # check if the current module has reset_parameters and if it is reset the weight
    reset_parameters = getattr(m, "reset_parameters", None)
    if callable(reset_parameters):
        m.reset_parameters()


def get_module_weights_sum(mdl: nn.Module):
    dict_sums = {}
    for name, w in mdl.named_parameters():
        if "weight" in name:
            value = w.data.sum().item()
            dict_sums[name] = value
    return dict_sums


def load_audio(file_path):
    """Load the audio file normalized in [-1, 1]

    Return Shapes:
        - x: :math:`[1, T]`
    """
    x, sr = torchaudio.load(file_path)
    assert (x > 1).sum() + (x < -1).sum() == 0
    return x, sr


def _amp_to_db(x, C=1, clip_val=1e-5):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def _db_to_amp(x, C=1):
    return torch.exp(x) / C


def amp_to_db(magnitudes):
    output = _amp_to_db(magnitudes)
    return output


def db_to_amp(magnitudes):
    output = _db_to_amp(magnitudes)
    return output


def wav_to_spec(y, n_fft, hop_length, win_length, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec


def spec_to_mel(spec, n_fft, num_mels, sample_rate, fmin, fmax):
    """
    Args Shapes:
        - spec : :math:`[B,C,T]`

    Return Shapes:
        - mel : :math:`[B,C,T]`
    """
    global mel_basis
    dtype_device = str(spec.dtype) + "_" + str(spec.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=spec.dtype, device=spec.device)
    mel = torch.matmul(mel_basis[fmax_dtype_device], spec)
    mel = amp_to_db(mel)
    return mel


def wav_to_mel(y, n_fft, num_mels, sample_rate, hop_length, win_length, fmin, fmax, center=False):
    """
    Args Shapes:
        - y : :math:`[B, 1, T]`

    Return Shapes:
        - spec : :math:`[B,C,T]`
    """
    y = y.squeeze(1)

    if torch.min(y) < -1.0:
        print("min value is ", torch.min(y))
    if torch.max(y) > 1.0:
        print("max value is ", torch.max(y))

    global mel_basis, hann_window
    dtype_device = str(y.dtype) + "_" + str(y.device)
    fmax_dtype_device = str(fmax) + "_" + dtype_device
    wnsize_dtype_device = str(win_length) + "_" + dtype_device
    if fmax_dtype_device not in mel_basis:
        mel = librosa_mel_fn(sample_rate, n_fft, num_mels, fmin, fmax)
        mel_basis[fmax_dtype_device] = torch.from_numpy(mel).to(dtype=y.dtype, device=y.device)
    if wnsize_dtype_device not in hann_window:
        hann_window[wnsize_dtype_device] = torch.hann_window(win_length).to(dtype=y.dtype, device=y.device)

    y = torch.nn.functional.pad(
        y.unsqueeze(1),
        (int((n_fft - hop_length) / 2), int((n_fft - hop_length) / 2)),
        mode="reflect",
    )
    y = y.squeeze(1)

    spec = torch.stft(
        y,
        n_fft,
        hop_length=hop_length,
        win_length=win_length,
        window=hann_window[wnsize_dtype_device],
        center=center,
        pad_mode="reflect",
        normalized=False,
        onesided=True,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec

def compute_pitch(x: np.ndarray, sample_rate, hop_length, pitch_fmax=800.0) -> np.ndarray:
    """Compute pitch (f0) of a waveform using the same parameters used for computing melspectrogram.

    Args:
        x (np.ndarray): Waveform.

    Returns:
        np.ndarray: Pitch.
    """
    # align F0 length to the spectrogram length
    if len(x) % hop_length == 0:
        x = np.pad(x, (0, hop_length // 2), mode="reflect")

    f0, t = pw.dio(
        x.astype(np.double),
        fs=sample_rate,
        f0_ceil=pitch_fmax,
        frame_period=1000 * hop_length / sample_rate,
    )
    f0 = pw.stonemask(x.astype(np.double), f0, t, sample_rate)
    return f0

##############################
# DATASET
##############################

class VITSF0Dataset(F0Dataset):
    def __init__(self, config, *args, **kwargs):
        self.audio_config = config.audio
        super().__init__(*args, **kwargs)
    
    def compute_or_load(self, wav_file):
        """
        compute pitch and return a numpy array of pitch values
        """
        pitch_file = self.create_pitch_file_path(wav_file, self.cache_path)
        if not os.path.exists(pitch_file):
            pitch = self._compute_and_save_pitch(wav_file, self.audio_config.sample_rate, self.audio_config.hop_length, pitch_file)
        else:
            pitch = np.load(pitch_file)
        return pitch.astype(np.float32)

    @staticmethod
    def _compute_and_save_pitch(wav_file, sample_rate, hop_length, pitch_file=None):
        wav, _ = load_audio(wav_file)
        pitch = compute_pitch(wav.squeeze().numpy(), sample_rate, hop_length)
        if pitch_file:
            np.save(pitch_file, pitch)
        return pitch
    


class VitsDataset(TTSDataset):
    def __init__(self, model_args, config, compute_pitch=False, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.model_args = model_args
        self.compute_pitch = compute_pitch
        self.use_precomputed_alignments = model_args.use_precomputed_alignments
        self.alignments_cache_path = model_args.alignments_cache_path

        if self.compute_pitch:
            self.f0_dataset = VITSF0Dataset(config,
                samples=self.samples, ap=self.ap, cache_path=self.f0_cache_path, precompute_num_workers=self.precompute_num_workers
            )

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        if self.model_args.encoder_sample_rate is not None:
            if wav.size(1) % self.model_args.encoder_sample_rate != 0:
                wav = wav[:, : -int(wav.size(1) % self.model_args.encoder_sample_rate)]

        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"])

        # get f0 values
        f0 = None
        if self.compute_pitch:
            f0 = self.get_f0(idx)["f0"]

        alignments = None
        if self.use_precomputed_alignments:
            align_file = os.path.join(self.alignments_cache_path, os.path.splitext(wav_filename)[0] + ".npy")
            alignments = self.get_attn_mask(align_file)

        # after phonemization the text length may change
        # this is a shameful 🤭 hack to prevent longer phonemes
        # TODO: find a better fix
        if len(token_ids) > self.max_text_len or wav.shape[1] < self.min_audio_len:
            self.rescue_item_idx += 1
            return self.__getitem__(self.rescue_item_idx)

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
            "pitch": f0,
            "alignments": alignments,

        }

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
            - waveform: :math:`[B, 1, T]`
            - waveform_lens: :math:`[B]`
            - waveform_rel_lens: :math:`[B]`
            - speaker_names: :math:`[B]`
            - language_names: :math:`[B]`
            - audiofile_paths: :math:`[B]`
            - raw_texts: :math:`[B]`
        """
        # convert list of dicts to dict of lists
        B = len(batch)
        batch = {k: [dic[k] for dic in batch] for k in batch[0]}

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x.size(1) for x in batch["wav"]]), dim=0, descending=True
        )

        max_text_len = max([len(x) for x in batch["token_ids"]])
        token_lens = torch.LongTensor(batch["token_len"])
        token_rel_lens = token_lens / token_lens.max()

        wav_lens = [w.shape[1] for w in batch["wav"]]
        wav_lens = torch.LongTensor(wav_lens)
        wav_lens_max = torch.max(wav_lens)
        wav_rel_lens = wav_lens / wav_lens_max

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)
        

        # format F0
        if self.compute_pitch:
            pitch = prepare_data(batch["pitch"])
            pitch = torch.FloatTensor(pitch)[:, None, :].contiguous() # B x 1 xT
        else:
            pitch = None
        
        padded_alignments = None
        if self.use_precomputed_alignments:
            alignments = batch["alignments"]
            max_len_1 = max((x.shape[0] for x in alignments))
            max_len_2 = max((x.shape[1] for x in alignments))
            padded_alignments = []
            for x in alignments:
                padded_alignment = np.pad(x, ((0, max_len_1 - x.shape[0]), (0, max_len_2 - x.shape[1])), mode="constant", constant_values=0)
                padded_alignments.append(padded_alignment)
            
            padded_alignments = torch.FloatTensor(np.stack(padded_alignments)).unsqueeze(1)

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "pitch": pitch,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "alignments": padded_alignments,
        }


##############################
# MODEL DEFINITION
##############################

@dataclass
class VitsArgs(Coqpit):
    """VITS model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels of the decoder. Defaults to 513.

        spec_segment_size (int):
            Decoder input segment size. Defaults to 32 `(32 * hoplength = waveform length)`.

        hidden_channels (int):
            Number of hidden channels of the model. Defaults to 192.

        hidden_channels_ffn_text_encoder (int):
            Number of hidden channels of the feed-forward layers of the text encoder transformer. Defaults to 256.

        num_heads_text_encoder (int):
            Number of attention heads of the text encoder transformer. Defaults to 2.

        num_layers_text_encoder (int):
            Number of transformer layers in the text encoder. Defaults to 6.

        kernel_size_text_encoder (int):
            Kernel size of the text encoder transformer FFN layers. Defaults to 3.

        dropout_p_text_encoder (float):
            Dropout rate of the text encoder. Defaults to 0.1.

        dropout_p_duration_predictor (float):
            Dropout rate of the duration predictor. Defaults to 0.1.

        kernel_size_posterior_encoder (int):
            Kernel size of the posterior encoder's WaveNet layers. Defaults to 5.

        dilatation_posterior_encoder (int):
            Dilation rate of the posterior encoder's WaveNet layers. Defaults to 1.

        num_layers_posterior_encoder (int):
            Number of posterior encoder's WaveNet layers. Defaults to 16.

        kernel_size_flow (int):
            Kernel size of the Residual Coupling layers of the flow network. Defaults to 5.

        dilatation_flow (int):
            Dilation rate of the Residual Coupling WaveNet layers of the flow network. Defaults to 1.

        num_layers_flow (int):
            Number of Residual Coupling WaveNet layers of the flow network. Defaults to 6.

        resblock_type_decoder (str):
            Type of the residual block in the decoder network. Defaults to "1".

        resblock_kernel_sizes_decoder (List[int]):
            Kernel sizes of the residual blocks in the decoder network. Defaults to `[3, 7, 11]`.

        resblock_dilation_sizes_decoder (List[List[int]]):
            Dilation sizes of the residual blocks in the decoder network. Defaults to `[[1, 3, 5], [1, 3, 5], [1, 3, 5]]`.

        upsample_rates_decoder (List[int]):
            Upsampling rates for each concecutive upsampling layer in the decoder network. The multiply of these
            values must be equal to the kop length used for computing spectrograms. Defaults to `[8, 8, 2, 2]`.

        upsample_initial_channel_decoder (int):
            Number of hidden channels of the first upsampling convolution layer of the decoder network. Defaults to 512.

        upsample_kernel_sizes_decoder (List[int]):
            Kernel sizes for each upsampling layer of the decoder network. Defaults to `[16, 16, 4, 4]`.

        periods_multi_period_discriminator (List[int]):
            Periods values for Vits Multi-Period Discriminator. Defaults to `[2, 3, 5, 7, 11]`.

        use_sdp (bool):
            Use Stochastic Duration Predictor. Defaults to True.

        noise_scale (float):
            Noise scale used for the sample noise tensor in training. Defaults to 1.0.

        inference_noise_scale (float):
            Noise scale used for the sample noise tensor in inference. Defaults to 0.667.

        length_scale (float):
            Scale factor for the predicted duration values. Smaller values result faster speech. Defaults to 1.

        noise_scale_dp (float):
            Noise scale used by the Stochastic Duration Predictor sample noise in training. Defaults to 1.0.

        inference_noise_scale_dp (float):
            Noise scale for the Stochastic Duration Predictor in inference. Defaults to 0.8.

        max_inference_len (int):
            Maximum inference length to limit the memory use. Defaults to None.

        init_discriminator (bool):
            Initialize the disciminator network if set True. Set False for inference. Defaults to True.

        use_spectral_norm_disriminator (bool):
            Use spectral normalization over weight norm in the discriminator. Defaults to False.

        use_speaker_embedding (bool):
            Enable/Disable speaker embedding for multi-speaker models. Defaults to False.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        speakers_file (str):
            Path to the speaker mapping file for the Speaker Manager. Defaults to None.

        speaker_embedding_channels (int):
            Number of speaker embedding channels. Defaults to 256.

        use_d_vector_file (bool):
            Enable/Disable the use of d-vectors for multi-speaker training. Defaults to False.

        d_vector_file (str):
            Path to the file including pre-computed speaker embeddings. Defaults to None.

        d_vector_dim (int):
            Number of d-vector channels. Defaults to 0.

        detach_dp_input (bool):
            Detach duration predictor's input from the network for stopping the gradients. Defaults to True.

        use_language_embedding (bool):
            Enable/Disable language embedding for multilingual models. Defaults to False.

        embedded_language_dim (int):
            Number of language embedding channels. Defaults to 4.

        num_languages (int):
            Number of languages for the language embedding layer. Defaults to 0.

        language_ids_file (str):
            Path to the language mapping file for the Language Manager. Defaults to None.

        use_speaker_encoder_as_loss (bool):
            Enable/Disable Speaker Consistency Loss (SCL). Defaults to False.

        encoder_config_path (str):
            Path to the file speaker encoder config file, to use for SCL. Defaults to "".

        encoder_model_path (str):
            Path to the file speaker encoder checkpoint file, to use for SCL. Defaults to "".

        use_pitch (bool):
            Use pitch predictor to learn the pitch. Defaults to False.

        pitch_predictor_hidden_channels (int):
            Number of hidden channels in the pitch predictor. Defaults to 256.

        pitch_predictor_dropout_p (float):
            Dropout rate for the pitch predictor. Defaults to 0.1.

        pitch_predictor_kernel_size (int):
            Kernel size of conv layers in the pitch predictor. Defaults to 3.

        pitch_embedding_kernel_size (int):
            Kernel size of the projection layer in the pitch predictor. Defaults to 3.

        condition_dp_on_speaker (bool):
            Condition the duration predictor on the speaker embedding. Defaults to True.

        freeze_encoder (bool):
            Freeze the encoder weigths during training. Defaults to False.

        freeze_DP (bool):
            Freeze the duration predictor weigths during training. Defaults to False.

        freeze_PE (bool):
            Freeze the posterior encoder weigths during training. Defaults to False.

        freeze_flow_encoder (bool):
            Freeze the flow encoder weigths during training. Defaults to False.

        freeze_waveform_decoder (bool):
            Freeze the waveform decoder weigths during training. Defaults to False.

        encoder_sample_rate (int):
            If not None this sample rate will be used for training the Posterior Encoder,
            flow, text_encoder and duration predictor. The decoder part (vocoder) will be
            trained with the `config.audio.sample_rate`. Defaults to None.

        interpolate_z (bool):
            If `encoder_sample_rate` not None and  this parameter True the nearest interpolation
            will be used to upsampling the latent variable z with the sampling rate `encoder_sample_rate`
            to the `config.audio.sample_rate`. If it is False you will need to add extra
            `upsample_rates_decoder` to match the shape. Defaults to True.

    """

    num_chars: int = 100
    out_channels: int = 513
    spec_segment_size: int = 32
    hidden_channels: int = 192
    hidden_channels_ffn_text_encoder: int = 768
    num_heads_text_encoder: int = 2
    num_layers_text_encoder: int = 6
    kernel_size_text_encoder: int = 3
    dropout_p_text_encoder: float = 0.1
    dropout_p_duration_predictor: float = 0.5
    kernel_size_posterior_encoder: int = 5
    dilation_rate_posterior_encoder: int = 1
    num_layers_posterior_encoder: int = 16
    kernel_size_flow: int = 5
    dilation_rate_flow: int = 1
    num_layers_flow: int = 4
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    periods_multi_period_discriminator: List[int] = field(default_factory=lambda: [2, 3, 5, 7, 11])
    use_sdp: bool = True
    noise_scale: float = 1.0
    inference_noise_scale: float = 0.667
    length_scale: float = 1
    noise_scale_dp: float = 1.0
    inference_noise_scale_dp: float = 1.0
    max_inference_len: int = None
    init_discriminator: bool = True
    use_spectral_norm_disriminator: bool = False
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 256
    use_d_vector_file: bool = False
    d_vector_dim: int = 0

    # use emotion embeddings
    use_emotion_embedding: bool = False
    use_external_emotions_embeddings: bool = False
    emotions_ids_file: str = None
    external_emotions_embs_file: str = None
    emotion_embedding_dim: int = 0
    num_emotions: int = 0
    use_text_enc_spk_reversal_classifier: bool = False
    use_text_enc_emo_classifier: bool = False

    # emotion and speaker embedding squeezer
    use_emotion_embedding_squeezer: bool = False
    emotion_embedding_squeezer_input_dim: int = 0
    use_speaker_embedding_squeezer: bool = False
    speaker_embedding_squeezer_input_dim: int = 0

    use_speaker_embedding_as_emotion: bool = False

    # prosody encoder
    use_prosody_encoder: bool = False
    prosody_encoder_type: str = "gst"
    detach_prosody_enc_input: bool = False
    condition_pros_enc_on_speaker: bool = False

    prosody_embedding_dim: int = 0
    prosody_encoder_num_heads: int = 1
    prosody_encoder_num_tokens: int = 5
    use_prosody_encoder_z_p_input: bool = False
    use_prosody_enc_spk_reversal_classifier: bool = False
    use_prosody_enc_emo_classifier: bool = False

    use_latent_discriminator: bool = False

    use_encoder_conditional_module: bool = False
    conditional_module_type: str = "fftransformer"
    conditional_module_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 3, "dropout_p": 0.1}
    )

    # Pitch predictor
    use_pitch: bool = False
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    detach_pp_input: bool = False
    use_precomputed_alignments: bool = False
    alignments_cache_path: str = ""
    pitch_mean: float = 0.0
    pitch_std: float = 0.0

    use_z_decoder: bool = False
    z_decoder_type: str = "fftransformer"
    z_decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )

    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    use_emotion_encoder_as_loss: bool = False
    encoder_config_path: str = ""
    encoder_model_path: str = ""
    condition_dp_on_speaker: bool = True
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False
    encoder_sample_rate: int = None
    interpolate_z: bool = True
    reinit_DP: bool = False
    reinit_text_encoder: bool = False


class Vits(BaseTTS):
    """VITS TTS model

    Paper::
        https://arxiv.org/pdf/2106.06103.pdf

    Paper Abstract::
        Several recent end-to-end text-to-speech (TTS) models enabling single-stage training and parallel
        sampling have been proposed, but their sample quality does not match that of two-stage TTS systems.
        In this work, we present a parallel endto-end TTS method that generates more natural sounding audio than
        current two-stage models. Our method adopts variational inference augmented with normalizing flows and
        an adversarial training process, which improves the expressive power of generative modeling. We also propose a
        stochastic duration predictor to synthesize speech with diverse rhythms from input text. With the
        uncertainty modeling over latent variables and the stochastic duration predictor, our method expresses the
        natural one-to-many relationship in which a text input can be spoken in multiple ways
        with different pitches and rhythms. A subjective human evaluation (mean opinion score, or MOS)
        on the LJ Speech, a single speaker dataset, shows that our method outperforms the best publicly
        available TTS systems and achieves a MOS comparable to ground truth.

    Check :class:`TTS.tts.configs.vits_config.VitsConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs.vits_config import VitsConfig
        >>> from TTS.tts.models.vits import Vits
        >>> config = VitsConfig()
        >>> model = Vits(config)
    """

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
        emotion_manager: EmotionManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)

        self.init_multispeaker(config)
        self.init_multilingual(config)
        self.init_upsampling()
        self.init_emotion(emotion_manager)
        self.init_consistency_loss()

        self.length_scale = self.args.length_scale
        self.noise_scale = self.args.noise_scale
        self.inference_noise_scale = self.args.inference_noise_scale
        self.inference_noise_scale_dp = self.args.inference_noise_scale_dp
        self.noise_scale_dp = self.args.noise_scale_dp
        self.max_inference_len = self.args.max_inference_len
        self.spec_segment_size = self.args.spec_segment_size

        self.text_encoder = TextEncoder(
            self.args.num_chars,
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.hidden_channels_ffn_text_encoder,
            self.args.num_heads_text_encoder,
            self.args.num_layers_text_encoder,
            self.args.kernel_size_text_encoder,
            self.args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
            emotion_emb_dim=self.args.emotion_embedding_dim,
            prosody_emb_dim=self.args.prosody_embedding_dim if not self.args.use_encoder_conditional_module and not self.args.use_z_decoder else 0
        )

        self.posterior_encoder = PosteriorEncoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_posterior_encoder,
            dilation_rate=self.args.dilation_rate_posterior_encoder,
            num_layers=self.args.num_layers_posterior_encoder,
            cond_channels=self.cond_embedding_dim,
        )

        self.flow = ResidualCouplingBlocks(
            self.args.hidden_channels,
            self.args.hidden_channels,
            kernel_size=self.args.kernel_size_flow,
            dilation_rate=self.args.dilation_rate_flow,
            num_layers=self.args.num_layers_flow,
            cond_channels=self.cond_embedding_dim,
        )

        dp_cond_embedding_dim = self.cond_embedding_dim if self.args.condition_dp_on_speaker else 0

        if self.args.use_emotion_embedding or self.args.use_external_emotions_embeddings:
            dp_cond_embedding_dim += self.args.emotion_embedding_dim

        if self.args.use_prosody_encoder and not self.args.use_encoder_conditional_module and not self.args.use_z_decoder:
            dp_cond_embedding_dim += self.args.prosody_embedding_dim

        dp_extra_inp_dim = 0
        if (
            self.args.use_emotion_embedding
            or self.args.use_external_emotions_embeddings
            or self.args.use_speaker_embedding_as_emotion
        ):
            dp_extra_inp_dim += self.args.emotion_embedding_dim

        if self.args.use_prosody_encoder and not self.args.use_encoder_conditional_module and not self.args.use_z_decoder:
            dp_extra_inp_dim += self.args.prosody_embedding_dim

        if self.args.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                self.args.hidden_channels + dp_extra_inp_dim,
                192,
                3,
                self.args.dropout_p_duration_predictor,
                4,
                cond_channels=dp_cond_embedding_dim,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                self.args.hidden_channels + dp_extra_inp_dim,
                256,
                3,
                self.args.dropout_p_duration_predictor,
                cond_channels=dp_cond_embedding_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        if self.args.use_z_decoder:
            self.z_decoder = forwardDecoder(
                self.args.hidden_channels,
                self.args.hidden_channels + self.cond_embedding_dim,
                self.args.z_decoder_type,
                self.args.z_decoder_params,
            )

        if self.args.use_encoder_conditional_module:
            self.encoder_conditional_module = forwardDecoder(
                self.args.hidden_channels,
                self.args.hidden_channels,
                self.args.conditional_module_type,
                self.args.conditional_module_params,
            )

        if self.args.use_pitch:
            if not self.args.use_encoder_conditional_module and not self.args.use_z_decoder:
                raise RuntimeError(
                    f" [!] use_pitch True is useless when use_encoder_conditional_module and use_z_decoder is False. Please active on of this conditional modules !!"
                )

            self.pitch_emb = nn.Conv1d(
                1,
                self.args.hidden_channels,
                kernel_size=self.args.pitch_predictor_kernel_size,
                padding=int((self.args.pitch_predictor_kernel_size - 1) / 2),
            )
            self.pitch_predictor = DurationPredictor(
                self.args.hidden_channels,
                self.args.pitch_predictor_hidden_channels,
                self.args.pitch_predictor_kernel_size,
                self.args.pitch_predictor_dropout_p,
                cond_channels=self.cond_embedding_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        if self.args.use_prosody_encoder:
            prosody_embedding_dim = self.args.prosody_embedding_dim if not self.args.use_encoder_conditional_module and not self.args.use_z_decoder else self.args.hidden_channels
            if self.args.prosody_encoder_type == "gst":
                self.prosody_encoder = VitsGST(
                    num_mel=self.args.hidden_channels,
                    num_heads=self.args.prosody_encoder_num_heads,
                    num_style_tokens=self.args.prosody_encoder_num_tokens,
                    gst_embedding_dim=prosody_embedding_dim,
                    embedded_speaker_dim=self.cond_embedding_dim if self.args.condition_pros_enc_on_speaker else None,
                )
            elif self.args.prosody_encoder_type == "vae":
                self.prosody_encoder = VitsVAE(
                    num_mel=self.args.hidden_channels,
                    capacitron_VAE_embedding_dim=prosody_embedding_dim,
                    speaker_embedding_dim=self.cond_embedding_dim if self.args.condition_pros_enc_on_speaker else None,
                )
            elif self.args.prosody_encoder_type == "resnet":
                self.prosody_encoder = ResNetProsodyEncoder(
                    input_dim=self.args.hidden_channels,
                    proj_dim=prosody_embedding_dim,
                    layers=[1, 2, 2, 1],
                    num_filters=[8, 16, 32, 64],
                    encoder_type="ASP",
                )

            else:
                raise RuntimeError(
                    f" [!] The Prosody encoder type {self.args.prosody_encoder_type} is not supported !!"
                )

            print(f" > Using the prosody Encoder type {self.args.prosody_encoder_type} with {len(list(self.prosody_encoder.parameters()))} trainable parameters !")

            if self.args.use_prosody_enc_spk_reversal_classifier:
                self.speaker_reversal_classifier = ReversalClassifier(
                    in_channels=self.args.prosody_embedding_dim,
                    out_channels=self.num_speakers,
                    hidden_channels=256,
                )
            if self.args.use_prosody_enc_emo_classifier:
                self.pros_enc_emotion_classifier = ReversalClassifier(
                    in_channels=self.args.prosody_embedding_dim,
                    out_channels=self.num_emotions,
                    hidden_channels=256,
                    reversal=False,
                )

        if self.args.use_emotion_embedding_squeezer:
            self.emotion_embedding_squeezer = nn.Linear(
                in_features=self.args.emotion_embedding_squeezer_input_dim, out_features=self.args.emotion_embedding_dim
            )

        if self.args.use_speaker_embedding_squeezer:
            self.speaker_embedding_squeezer = nn.Linear(
                in_features=self.args.speaker_embedding_squeezer_input_dim, out_features=self.cond_embedding_dim
            )

        if self.args.use_text_enc_spk_reversal_classifier:
            self.speaker_text_enc_reversal_classifier = ReversalClassifier(
                in_channels=self.args.hidden_channels + dp_extra_inp_dim,
                out_channels=self.num_speakers,
                hidden_channels=256,
            )

        if self.args.use_text_enc_emo_classifier:
            self.emo_text_enc_classifier = ReversalClassifier(
                in_channels=self.args.hidden_channels,
                out_channels=self.num_emotions,
                hidden_channels=256,
                reversal=False,
            )

        self.waveform_decoder = HifiganGenerator(
            self.args.hidden_channels,
            1,
            self.args.resblock_type_decoder,
            self.args.resblock_dilation_sizes_decoder,
            self.args.resblock_kernel_sizes_decoder,
            self.args.upsample_kernel_sizes_decoder,
            self.args.upsample_initial_channel_decoder,
            self.args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.cond_embedding_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(
                periods=self.args.periods_multi_period_discriminator,
                use_spectral_norm=self.args.use_spectral_norm_disriminator,
                use_latent_disc=self.args.use_latent_discriminator,
                hidden_channels=self.args.hidden_channels,
            )

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.cond_embedding_dim = 0
        self.num_speakers = self.args.num_speakers
        self.audio_transform = None

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

    def init_consistency_loss(self):
        if self.args.use_speaker_encoder_as_loss and self.args.use_emotion_encoder_as_loss:
            raise RuntimeError(
                " [!] The use of speaker consistency loss (SCL) and emotion consistency loss (ECL) together is not supported, please disable one of those !!"
            )

        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.encoder is None and (
                not self.args.encoder_model_path or not self.args.encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify encoder_model_path and encoder_config_path !!"
                )
            self.speaker_manager.encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.encoder, "audio_config")
                and self.config.audio["sample_rate"] != self.speaker_manager.encoder.audio_config["sample_rate"]
            ):
                # pylint: disable=W0101,W0105
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio["sample_rate"],
                    new_freq=self.speaker_manager.encoder.audio_config["sample_rate"],
                )

        elif self.args.use_emotion_encoder_as_loss:
            if self.emotion_manager.encoder is None and (
                not self.args.encoder_model_path or not self.args.encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the emotion consistency loss (ECL) you need to specify encoder_model_path and encoder_config_path !!"
                )

            self.emotion_manager.encoder.eval()
            print(" > External Emotion Encoder Loaded !!")

            if (
                hasattr(self.emotion_manager.encoder, "audio_config")
                and self.config.audio["sample_rate"] != self.emotion_manager.encoder.audio_config["sample_rate"]
            ):
                # pylint: disable=W0101,W0105
                self.audio_transform = torchaudio.transforms.Resample(
                    orig_freq=self.config.audio["sample_rate"],
                    new_freq=self.emotion_manager.encoder.audio_config["sample_rate"],
                )

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.cond_embedding_dim += self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.args.speaker_embedding_channels)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.cond_embedding_dim += self.args.d_vector_dim

    def init_multilingual(self, config: Coqpit):
        """Initialize multilingual modules of a model.

        Args:
            config (Coqpit): Model configuration.
        """
        if self.args.language_ids_file is not None:
            self.language_manager = LanguageManager(language_ids_file_path=config.language_ids_file)

        if self.args.use_language_embedding and self.language_manager:
            print(" > initialization of language-embedding layers.")
            self.num_languages = self.language_manager.num_languages
            self.embedded_language_dim = self.args.embedded_language_dim
            self.emb_l = nn.Embedding(self.num_languages, self.embedded_language_dim)
            torch.nn.init.xavier_uniform_(self.emb_l.weight)
        else:
            self.embedded_language_dim = 0

    def init_upsampling(self):
        """
        Initialize upsampling modules of a model.
        """
        if self.args.encoder_sample_rate:
            self.interpolate_factor = self.config.audio["sample_rate"] / self.args.encoder_sample_rate
            self.audio_resampler = torchaudio.transforms.Resample(
                orig_freq=self.config.audio["sample_rate"], new_freq=self.args.encoder_sample_rate
            )  # pylint: disable=W0201

    def on_init_end(self, trainer):  # pylint: disable=W0613
        """Reinit layes if needed"""
        if self.args.reinit_DP:
            before_dict = get_module_weights_sum(self.duration_predictor)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.duration_predictor.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.duration_predictor)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Duration Predictor was not reinit check it !")
            print(" > Duration Predictor was reinit.")

        if self.args.reinit_text_encoder:
            before_dict = get_module_weights_sum(self.text_encoder)
            # Applies weights_reset recursively to every submodule of the duration predictor
            self.text_encoder.apply(fn=weights_reset)
            after_dict = get_module_weights_sum(self.text_encoder)
            for key, value in after_dict.items():
                if value == before_dict[key]:
                    raise RuntimeError(" [!] The weights of Text Encoder was not reinit check it !")
            print(" > Text Encoder was reinit.")

    def init_emotion(self, emotion_manager: EmotionManager):
        # pylint: disable=attribute-defined-outside-init
        """Initialize emotion modules of a model. A model can be trained either with a emotion embedding layer
        or with external `embeddings` computed from a emotion encoder model.

        You must provide a `emotion_manager` at initialization to set up the emotion modules.

        Args:
            emotion_manager (Coqpit): Emotion Manager.
        """
        self.emotion_manager = emotion_manager
        self.num_emotions = self.args.num_emotions

        if self.emotion_manager:
            self.num_emotions = self.emotion_manager.num_emotions

        if self.args.use_emotion_embedding:
            if self.num_emotions > 0:
                print(" > initialization of emotion-embedding layers.")
                self.emb_emotion = nn.Embedding(self.num_emotions, self.args.emotion_embedding_dim)

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid, eid, eg, pf, ssid, ssg = self._set_cond_input(aux_input)
        return {
            "speaker_ids": sid,
            "style_wav": None,
            "d_vectors": g,
            "language_ids": lid,
            "emotion_embeddings": eg,
            "emotion_ids": eid,
            "style_feature": pf,
            "style_speaker_ids": ssid,
            "style_speaker_d_vectors": ssg,
        }

    def _freeze_layers(self):
        if self.args.freeze_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False

            if hasattr(self, "emb_l"):
                for param in self.emb_l.parameters():
                    param.requires_grad = False

        if self.args.freeze_PE:
            for param in self.posterior_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_DP:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_flow_decoder:
            for param in self.flow.parameters():
                param.requires_grad = False

        if self.args.freeze_waveform_decoder:
            for param in self.waveform_decoder.parameters():
                param.requires_grad = False

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid, eid, eg, pf, ssid, ssg = None, None, None, None, None, None, None, None
        if "speaker_ids" in aux_input and aux_input["speaker_ids"] is not None:
            sid = aux_input["speaker_ids"]
            if sid.ndim == 0:
                sid = sid.unsqueeze_(0)
        if "d_vectors" in aux_input and aux_input["d_vectors"] is not None:
            g = F.normalize(aux_input["d_vectors"]).unsqueeze(-1)
            if g.ndim == 2:
                g = g.unsqueeze_(0)

        if "language_ids" in aux_input and aux_input["language_ids"] is not None:
            lid = aux_input["language_ids"]
            if lid.ndim == 0:
                lid = lid.unsqueeze_(0)

        if "emotion_ids" in aux_input and aux_input["emotion_ids"] is not None:
            eid = aux_input["emotion_ids"]
            if eid.ndim == 0:
                eid = eid.unsqueeze_(0)

        if "emotion_embeddings" in aux_input and aux_input["emotion_embeddings"] is not None:
            eg = F.normalize(aux_input["emotion_embeddings"]).unsqueeze(-1)
            if eg.ndim == 2:
                eg = eg.unsqueeze_(0)

        if "style_feature" in aux_input and aux_input["style_feature"] is not None:
            pf = aux_input["style_feature"]
            if pf.ndim == 2:
                pf = pf.unsqueeze_(0)

        if "style_speaker_id" in aux_input and aux_input["style_speaker_id"] is not None:
            ssid = aux_input["style_speaker_id"]
            if ssid.ndim == 0:
                ssid = ssid.unsqueeze_(0)

        if "style_speaker_d_vector" in aux_input and aux_input["style_speaker_d_vector"] is not None:
            ssg = F.normalize(aux_input["style_speaker_d_vector"]).unsqueeze(-1)
            if ssg.ndim == 2:
                ssg = ssg.unsqueeze_(0)

        return sid, g, lid, eid, eg, pf, ssid, ssg

    def _set_speaker_input(self, aux_input: Dict):
        d_vectors = aux_input.get("d_vectors", None)
        speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_lengths: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
        g_pp: torch.IntTensor = None,
        pitch_transform: Callable=None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Pitch predictor forward pass.

        1. Predict pitch from encoder outputs.
        2. In training - Compute average pitch values for each input character from the ground truth pitch values.
        3. Embed average pitch values.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            x_mask (torch.IntTensor): Input sequence mask.
            pitch (torch.FloatTensor, optional): Ground truth pitch values. Defaults to None.
            dr (torch.IntTensor, optional): Ground truth durations. Defaults to None.
            g_pp (torch.IntTensor, optional): Speaker/prosody embedding to condition the pithc predictor. Defaults to None.
            pitch_transform (Callable, optional): Pitch transform function. Defaults to None.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """

        x_mask = torch.unsqueeze(sequence_mask(x_lengths, o_en.size(2)), 1).to(o_en.dtype)  # [b, 1, t]

        pred_avg_pitch = self.pitch_predictor(
            o_en, 
            x_mask,
            g=g_pp.detach() if self.args.detach_pp_input and g_pp is not None else g_pp
        )

        if pitch_transform is not None:
            pred_avg_pitch = pitch_transform(pred_avg_pitch, x_mask.sum(dim=(1,2)), self.args.pitch_mean, self.args.pitch_std)

        pitch_loss = None
        pred_avg_pitch_emb = None
        gt_avg_pitch_emb = None
        if pitch is not None:
            gt_avg_pitch = average_over_durations(pitch, dr.squeeze()).detach()
            pitch_loss = torch.sum(torch.sum((gt_avg_pitch - pred_avg_pitch) ** 2, [1, 2]) / torch.sum(x_mask))
            gt_avg_pitch_emb = self.pitch_emb(gt_avg_pitch)
        else:
            pred_avg_pitch_emb = self.pitch_emb(pred_avg_pitch)

        return pitch_loss, gt_avg_pitch_emb, pred_avg_pitch_emb

    def forward_mas(self, outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g, lang_emb):
        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p**2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t, t']

        # duration predictor
        attn_durations = attn.sum(3)
        if self.args.use_sdp:
            loss_duration = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                attn_durations,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = loss_duration / torch.sum(x_mask)
        else:
            attn_log_durations = torch.log(attn_durations + 1e-6) * x_mask
            log_durations = self.duration_predictor(
                x.detach() if self.args.detach_dp_input else x,
                x_mask,
                g=g.detach() if self.args.detach_dp_input and g is not None else g,
                lang_emb=lang_emb.detach() if self.args.detach_dp_input and lang_emb is not None else lang_emb,
            )
            loss_duration = torch.sum((log_durations - attn_log_durations) ** 2, [1, 2]) / torch.sum(x_mask)
        outputs["loss_duration"] = loss_duration
        return outputs, attn

    def upsampling_z(self, z, slice_ids=None, y_lengths=None, y_mask=None):
        spec_segment_size = self.spec_segment_size
        if self.args.encoder_sample_rate:
            # recompute the slices and spec_segment_size if needed
            slice_ids = slice_ids * int(self.interpolate_factor) if slice_ids is not None else slice_ids
            spec_segment_size = spec_segment_size * int(self.interpolate_factor)
            # interpolate z if needed
            if self.args.interpolate_z:
                z = torch.nn.functional.interpolate(z, scale_factor=[self.interpolate_factor], mode="linear").squeeze(0)
                # recompute the mask if needed
                if y_lengths is not None and y_mask is not None:
                    y_mask = (
                        sequence_mask(y_lengths * self.interpolate_factor, None).to(y_mask.dtype).unsqueeze(1)
                    )  # [B, 1, T_dec_resampled]

        return z, spec_segment_size, slice_ids, y_mask

    def forward(  # pylint: disable=dangerous-default-value
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        pitch: torch.tensor,
        alignments: torch.tensor,
        aux_input={
            "d_vectors": None,
            "speaker_ids": None,
            "language_ids": None,
            "emotion_embeddings": None,
            "emotion_ids": None,
        },
    ) -> Dict:
        """Forward pass of the model.

        Args:
            x (torch.tensor): Batch of input character sequence IDs.
            x_lengths (torch.tensor): Batch of input character sequence lengths.
            y (torch.tensor): Batch of input spectrograms.
            y_lengths (torch.tensor): Batch of input spectrogram lengths.
            waveform (torch.tensor): Batch of ground truth waveforms per sample.
            aux_input (dict, optional): Auxiliary inputs for multi-speaker and multi-lingual training.
                Defaults to {"d_vectors": None, "speaker_ids": None, "language_ids": None}.

        Returns:
            Dict: model outputs keyed by the output name.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - y: :math:`[B, C, T_spec]`
            - y_lengths: :math:`[B]`
            - waveform: :math:`[B, 1, T_wav]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
            - m_q: :math:`[B, C, T_dec]`
            - logs_q: :math:`[B, C, T_dec]`
            - waveform_seg: :math:`[B, 1, spec_seg_size * hop_length]`
            - gt_cons_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
            - syn_cons_emb: :math:`[B, 1, speaker_encoder.proj_dim]`
        """
        outputs = {}
        sid, g, lid, eid, eg, _, _, _ = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # emotion embedding
        if self.args.use_emotion_embedding and eid is not None and eg is None:
            eg = self.emb_emotion(eid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        if self.args.use_speaker_embedding_as_emotion:
            eg = g

        # squeezers
        if self.args.use_emotion_embedding_squeezer:
            if (
                self.args.use_emotion_embedding
                or self.args.use_external_emotions_embeddings
                or self.args.use_speaker_embedding_as_emotion
            ):
                eg = F.normalize(self.emotion_embedding_squeezer(eg.squeeze(-1))).unsqueeze(-1)

        if self.args.use_speaker_embedding_squeezer:
            if self.args.use_speaker_embedding or self.args.use_d_vector_file:
                g = F.normalize(self.speaker_embedding_squeezer(g.squeeze(-1))).unsqueeze(-1)

        # duration predictor
        g_dp = g if self.args.condition_dp_on_speaker else None
        if eg is not None and (self.args.use_emotion_embedding or self.args.use_external_emotions_embeddings):
            if g_dp is None:
                g_dp = eg
            else:
                g_dp = torch.cat([g_dp, eg], dim=1)  # [b, h1+h2, 1]

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # prosody embedding
        pros_emb = None
        vae_outputs = None
        l_pros_speaker = None
        l_pros_emotion = None
        if self.args.use_prosody_encoder:
            prosody_encoder_input = z_p if self.args.use_prosody_encoder_z_p_input else z
            pros_emb, vae_outputs = self.prosody_encoder(
                prosody_encoder_input.detach() if self.args.detach_prosody_enc_input else prosody_encoder_input,
                y_lengths,
                speaker_embedding=g if self.args.condition_pros_enc_on_speaker else None
            )

            pros_emb = pros_emb.transpose(1, 2)

            if self.args.use_prosody_enc_spk_reversal_classifier:
                _, l_pros_speaker = self.speaker_reversal_classifier(pros_emb.transpose(1, 2), sid, x_mask=None)
            if self.args.use_prosody_enc_emo_classifier:
                _, l_pros_emotion = self.pros_enc_emotion_classifier(pros_emb.transpose(1, 2), eid, x_mask=None)

        x, m_p, logs_p, x_mask = self.text_encoder(
            x,
            x_lengths,
            lang_emb=lang_emb,
            emo_emb=eg,
            pros_emb=pros_emb if not self.args.use_encoder_conditional_module and not self.args.use_z_decoder else None
        )

        # reversal speaker loss to force the encoder to be speaker identity free
        l_text_speaker = None
        if self.args.use_text_enc_spk_reversal_classifier:
            _, l_text_speaker = self.speaker_text_enc_reversal_classifier(x.transpose(1, 2), sid, x_mask=None)

        l_text_emotion = None
        # reversal speaker loss to force the encoder to be speaker identity free
        if self.args.use_text_enc_emo_classifier:
            _, l_text_emotion = self.emo_text_enc_classifier(m_p.transpose(1, 2), eid, x_mask=x_mask)

        # add prosody embedding on x if needed
        if self.args.use_prosody_encoder and (self.args.use_encoder_conditional_module or self.args.use_z_decoder):
            x = x + pros_emb.expand(-1, -1, x.size(2))

        # add prosody embedding when necessary
        if self.args.use_prosody_encoder and not self.args.use_encoder_conditional_module and not self.args.use_z_decoder:
            if g_dp is None:
                g_dp = pros_emb
            else:
                g_dp = torch.cat([g_dp, pros_emb], dim=1)  # [b, h1+h2, 1]

        outputs, attn = self.forward_mas(outputs, z_p, m_p, logs_p, x, x_mask, y_mask, g=g_dp, lang_emb=lang_emb)

        # add pitch
        pitch_loss = None
        gt_avg_pitch_emb = None
        if self.args.use_pitch:
            pitch_loss, gt_avg_pitch_emb, _ = self.forward_pitch_predictor(x, x_lengths, pitch, attn.sum(3), g)
            x = x + gt_avg_pitch_emb
            print(gt_avg_pitch_emb.shape, x.shape)

        z_p_avg = None
        if self.args.use_latent_discriminator:
            # average the z_p for the latent discriminator
            z_p_avg = average_over_durations(z_p, attn.sum(3).squeeze())

        # conditional module
        conditional_module_loss = None
        new_m_p = None
        if self.args.use_encoder_conditional_module:
            new_m_p = self.encoder_conditional_module(x, x_mask) * x_mask
            if z_p_avg is None:
                z_p_avg = average_over_durations(z_p, attn.sum(3).squeeze()).detach()
            else:
                z_p_avg = z_p_avg.detach()

            conditional_module_loss = torch.nn.functional.l1_loss(new_m_p, z_p_avg)

        # expand prior
        m_p_expanded = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p_expanded = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # z decoder
        z_decoder_loss = None
        if self.args.use_z_decoder:
            dec_input = torch.einsum("klmn, kjm -> kjn", [attn, x])
            # add speaker emb
            if g is not None:
                dec_input = torch.cat((dec_input, g.expand(-1, -1, dec_input.size(2))), dim=1)

            # decoder pass
            z_decoder = self.z_decoder(dec_input, y_mask, g=None)
            z_decoder_loss = torch.nn.functional.l1_loss(z_decoder * y_mask, z)

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size, let_short_samples=True, pad_short=True)

        # interpolate z if needed
        z_slice, spec_segment_size, slice_ids, _ = self.upsampling_z(z_slice, slice_ids=slice_ids)

        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )

        if self.args.use_speaker_encoder_as_loss or self.args.use_emotion_encoder_as_loss:
            encoder = (
                self.speaker_manager.encoder if self.args.use_speaker_encoder_as_loss else self.emotion_manager.encoder
            )
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            if next(encoder.parameters()).device != wavs_batch.device:
                encoder = encoder.to(wavs_batch.device)

            pred_embs = encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_cons_emb, syn_cons_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_cons_emb, syn_cons_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "m_p_unexpanded": m_p if new_m_p is None else new_m_p,
                "z_p_avg": z_p_avg,
                "m_p": m_p_expanded,
                "logs_p": logs_p_expanded,
                "z": z,
                "z_p": z_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_cons_emb": gt_cons_emb,
                "syn_cons_emb": syn_cons_emb,
                "slice_ids": slice_ids,
                "vae_outputs": vae_outputs,
                "loss_prosody_enc_spk_rev_classifier": l_pros_speaker,
                "loss_prosody_enc_emo_classifier": l_pros_emotion,
                "loss_text_enc_spk_rev_classifier": l_text_speaker,
                "loss_text_enc_emo_classifier": l_text_emotion,
                "pitch_loss": pitch_loss,
                "z_decoder_loss": z_decoder_loss,
                "conditional_module_loss": conditional_module_loss,
            }
        )
        return outputs

    @staticmethod
    def _set_x_lengths(x, aux_input):
        if "x_lengths" in aux_input and aux_input["x_lengths"] is not None:
            return aux_input["x_lengths"]
        return torch.tensor(x.shape[1:2]).to(x.device)

    @torch.no_grad()
    def inference(
        self,
        x,
        aux_input={
            "x_lengths": None,
            "d_vectors": None,
            "speaker_ids": None,
            "language_ids": None,
            "emotion_embeddings": None,
            "emotion_ids": None,
            "style_feature": None,
        },
        pitch_transform=None,
    ):  # pylint: disable=dangerous-default-value
        """
        Note:
            To run in batch mode, provide `x_lengths` else model assumes that the batch size is 1.

        Shapes:
            - x: :math:`[B, T_seq]`
            - x_lengths: :math:`[B]`
            - d_vectors: :math:`[B, C]`
            - speaker_ids: :math:`[B]`

        Return Shapes:
            - model_outputs: :math:`[B, 1, T_wav]`
            - alignments: :math:`[B, T_seq, T_dec]`
            - z: :math:`[B, C, T_dec]`
            - z_p: :math:`[B, C, T_dec]`
            - m_p: :math:`[B, C, T_dec]`
            - logs_p: :math:`[B, C, T_dec]`
        """
        sid, g, lid, eid, eg, pf, ssid, ssg = self._set_cond_input(aux_input)
        x_lengths = self._set_x_lengths(x, aux_input)

        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # emotion embedding
        if self.args.use_emotion_embedding and eid is not None and eg is None:
            eg = self.emb_emotion(eid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        if self.args.use_speaker_embedding_as_emotion:
            eg = g

        # squeezers
        if self.args.use_emotion_embedding_squeezer:
            if (
                self.args.use_emotion_embedding
                or self.args.use_external_emotions_embeddings
                or self.args.use_speaker_embedding_as_emotion
            ):
                eg = F.normalize(self.emotion_embedding_squeezer(eg.squeeze(-1))).unsqueeze(-1)

        if self.args.use_speaker_embedding_squeezer:
            if self.args.use_speaker_embedding or self.args.use_d_vector_file:
                g = F.normalize(self.speaker_embedding_squeezer(g.squeeze(-1))).unsqueeze(-1)

        # prosody embedding
        pros_emb = None
        if self.args.use_prosody_encoder:
            # speaker embedding for the style speaker
            if self.args.use_speaker_embedding and ssid is not None:
                ssg = self.emb_g(ssid).unsqueeze(-1)

            # extract posterior encoder feature
            pf_lengths = torch.tensor([pf.size(-1)]).to(pf.device)
            z_pro, _, _, z_pro_y_mask = self.posterior_encoder(pf, pf_lengths, g=ssg)
            if not self.args.use_prosody_encoder_z_p_input:
                pros_emb, _ = self.prosody_encoder(z_pro, pf_lengths, speaker_embedding=ssg if self.args.condition_pros_enc_on_speaker else None)
            else:
                z_p_inf = self.flow(z_pro, z_pro_y_mask, g=ssg)
                pros_emb, _ = self.prosody_encoder(z_p_inf, pf_lengths, speaker_embedding=ssg if self.args.condition_pros_enc_on_speaker else None)             

            pros_emb = pros_emb.transpose(1, 2)

        x, m_p, logs_p, x_mask = self.text_encoder(
            x,
            x_lengths,
            lang_emb=lang_emb,
            emo_emb=eg,
            pros_emb=pros_emb if not self.args.use_encoder_conditional_module and not self.args.use_z_decoder else None
        )

        # add prosody embedding on x if needed
        if self.args.use_prosody_encoder and (self.args.use_encoder_conditional_module or self.args.use_z_decoder):
            x = x + pros_emb.expand(-1, -1, x.size(2))

        # duration predictor
        g_dp = g if self.args.condition_dp_on_speaker else None
        if eg is not None and (self.args.use_emotion_embedding or self.args.use_external_emotions_embeddings):
            if g_dp is None:
                g_dp = eg
            else:
                g_dp = torch.cat([g_dp, eg], dim=1)  # [b, h1+h2, 1]

        if self.args.use_prosody_encoder and not self.args.use_encoder_conditional_module and not self.args.use_z_decoder:
            if g_dp is None:
                g_dp = pros_emb
            else:
                g_dp = torch.cat([g_dp, pros_emb], dim=1)  # [b, h1+h2, 1]

        if self.args.use_sdp:
            logw = self.duration_predictor(
                x,
                x_mask,
                g=g_dp,
                reverse=True,
                noise_scale=self.inference_noise_scale_dp,
                lang_emb=lang_emb,
            )
        else:
            logw = self.duration_predictor(x, x_mask, g=g_dp, lang_emb=lang_emb)

        w = torch.exp(logw) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype).unsqueeze(1)  # [B, 1, T_dec]

        attn_mask = x_mask * y_mask.transpose(1, 2)  # [B, 1, T_enc] * [B, T_dec, 1]
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        if self.args.use_pitch:
            _, _, pred_avg_pitch_emb = self.forward_pitch_predictor(x, x_lengths, g_pp=g, pitch_transform=pitch_transform)
            x = x + pred_avg_pitch_emb

        if self.args.use_encoder_conditional_module:
            m_p = self.encoder_conditional_module(x, x_mask)

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale

        if self.args.use_z_decoder:
            dec_input = torch.matmul(attn.transpose(1, 2), x.transpose(1, 2)).transpose(1, 2)

            # add speaker emb
            if g is not None:
                dec_input = torch.cat((dec_input, g.expand(-1, -1, dec_input.size(2))), dim=1)

            # decoder pass
            z = self.z_decoder(dec_input, y_mask, g=None)
        else:
            z = self.flow(z_p, y_mask, g=g, reverse=True)

        # upsampling if needed
        z, _, _, y_mask = self.upsampling_z(z, y_lengths=y_lengths, y_mask=y_mask)

        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {
            "model_outputs": o,
            "alignments": attn.squeeze(1),
            "durations": w_ceil,
            "z": z,
            "z_p": z_p,
            "m_p": m_p,
            "logs_p": logs_p,
            "y_mask": y_mask,
            "pitch": pred_avg_pitch_emb,
        }
        return outputs

    def compute_style_feature(self, style_wav_path):
        style_wav, sr = torchaudio.load(style_wav_path)
        if sr != self.config.audio.sample_rate and self.args.encoder_sample_rate is None:
            raise RuntimeError(
                f" [!] Style reference need to have sampling rate equal to {self.config.audio.sample_rate} !!"
            )
        elif self.args.encoder_sample_rate is not None and sr != self.args.encoder_sample_rate:
            raise RuntimeError(
                f" [!] Style reference need to have sampling rate equal to {self.args.encoder_sample_rate} !!"
            )
        y = wav_to_spec(
            style_wav.unsqueeze(1),
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        )
        return y

    @torch.no_grad()
    def inference_voice_conversion(
        self, reference_wav, speaker_id=None, d_vector=None, reference_speaker_id=None, reference_d_vector=None
    ):
        """Inference for voice conversion

        Args:
            reference_wav (Tensor): Reference wavform. Tensor of shape [B, T]
            speaker_id (Tensor): speaker_id of the target speaker. Tensor of shape [B]
            d_vector (Tensor): d_vector embedding of target speaker. Tensor of shape `[B, C]`
            reference_speaker_id (Tensor): speaker_id of the reference_wav speaker. Tensor of shape [B]
            reference_d_vector (Tensor): d_vector embedding of the reference_wav speaker. Tensor of shape `[B, C]`
        """
        # compute spectrograms
        y = wav_to_spec(
            reference_wav,
            self.config.audio.fft_size,
            self.config.audio.hop_length,
            self.config.audio.win_length,
            center=False,
        )
        y_lengths = torch.tensor([y.size(-1)]).to(y.device)
        speaker_cond_src = reference_speaker_id if reference_speaker_id is not None else reference_d_vector
        speaker_cond_tgt = speaker_id if speaker_id is not None else d_vector
        # print(y.shape, y_lengths.shape)
        wav, _, _ = self.voice_conversion(y, y_lengths, speaker_cond_src, speaker_cond_tgt)
        return wav

    def voice_conversion(self, y, y_lengths, speaker_cond_src, speaker_cond_tgt):
        """Forward pass for voice conversion

        TODO: create an end-point for voice conversion

        Args:
            y (Tensor): Reference spectrograms. Tensor of shape [B, T, C]
            y_lengths (Tensor): Length of each reference spectrogram. Tensor of shape [B]
            speaker_cond_src (Tensor): Reference speaker ID. Tensor of shape [B,]
            speaker_cond_tgt (Tensor): Target speaker ID. Tensor of shape [B,]
        """
        assert self.num_speakers > 0, "num_speakers have to be larger than 0."
        # speaker embedding
        if self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            g_src = self.emb_g(speaker_cond_src).unsqueeze(-1)
            g_tgt = self.emb_g(speaker_cond_tgt).unsqueeze(-1)
        elif not self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y, y_lengths, g=g_src)
        z_p = self.flow(z, y_mask, g=g_src)
        z_hat = self.flow(z_p, y_mask, g=g_tgt, reverse=True)
        o_hat = self.waveform_decoder(z_hat * y_mask, g=g_tgt)
        return o_hat, y_mask, (z, z_p, z_hat)

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        """Perform a single training step. Run the model forward pass and compute losses.

        Args:
            batch (Dict): Input tensors.
            criterion (nn.Module): Loss layer designed for the model.
            optimizer_idx (int): Index of optimizer to use. 0 for the generator and 1 for the discriminator networks.

        Returns:
            Tuple[Dict, Dict]: Model ouputs and computed losses.
        """

        self._freeze_layers()

        spec_lens = batch["spec_lens"]

        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lenghts = batch["token_lens"]
            spec = batch["spec"]

            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]
            emotion_embeddings = batch["emotion_embeddings"]
            emotion_ids = batch["emotion_ids"]
            waveform = batch["waveform"]
            pitch = batch["pitch"]
            alignments = batch["alignments"]

            # generator pass
            outputs = self.forward(
                tokens,
                token_lenghts,
                spec,
                spec_lens,
                waveform,
                pitch,
                alignments,
                aux_input={
                    "d_vectors": d_vectors,
                    "speaker_ids": speaker_ids,
                    "language_ids": language_ids,
                    "emotion_embeddings": emotion_embeddings,
                    "emotion_ids": emotion_ids,
                },
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_disc_fake, _, scores_disc_real, _, scores_disc_mp, _, scores_disc_zp, _ = self.disc(
                outputs["model_outputs"].detach(),
                outputs["waveform_seg"],
                outputs["m_p_unexpanded"].detach(), 
                outputs["z_p_avg"].detach() if outputs["z_p_avg"] is not None else None,
            )

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](scores_disc_real, scores_disc_fake, scores_disc_zp, scores_disc_mp)
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel"]

            # compute melspec segment
            with autocast(enabled=False):

                if self.args.encoder_sample_rate:
                    spec_segment_size = self.spec_segment_size * int(self.interpolate_factor)
                else:
                    spec_segment_size = self.spec_segment_size

                mel_slice = segment(
                    mel.float(), self.model_outputs_cache["slice_ids"], spec_segment_size, pad_short=True
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

            # compute discriminator scores and features
            (
                scores_disc_fake,
                feats_disc_fake,
                _,
                feats_disc_real,
                scores_disc_mp,
                feats_disc_mp,
                _,
                feats_disc_zp,
            ) = self.disc(
                self.model_outputs_cache["model_outputs"],
                self.model_outputs_cache["waveform_seg"],
                self.model_outputs_cache["m_p_unexpanded"],
                self.model_outputs_cache["z_p_avg"].detach() if self.model_outputs_cache["z_p_avg"] is not None else None,
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    mel_slice_hat=mel_slice.float(),
                    mel_slice=mel_slice_hat.float(),
                    z_p=self.model_outputs_cache["z_p"].float(),
                    logs_q=self.model_outputs_cache["logs_q"].float(),
                    m_p=self.model_outputs_cache["m_p"].float(),
                    logs_p=self.model_outputs_cache["logs_p"].float(),
                    z_len=spec_lens,
                    scores_disc_fake=scores_disc_fake,
                    feats_disc_fake=feats_disc_fake,
                    feats_disc_real=feats_disc_real,
                    loss_duration=self.model_outputs_cache["loss_duration"],
                    use_encoder_consistency_loss=self.args.use_speaker_encoder_as_loss
                    or self.args.use_emotion_encoder_as_loss,
                    gt_cons_emb=self.model_outputs_cache["gt_cons_emb"],
                    syn_cons_emb=self.model_outputs_cache["syn_cons_emb"],
                    vae_outputs=self.model_outputs_cache["vae_outputs"],
                    loss_prosody_enc_spk_rev_classifier=self.model_outputs_cache["loss_prosody_enc_spk_rev_classifier"],
                    loss_prosody_enc_emo_classifier=self.model_outputs_cache["loss_prosody_enc_emo_classifier"],
                    loss_text_enc_spk_rev_classifier=self.model_outputs_cache["loss_text_enc_spk_rev_classifier"],
                    loss_text_enc_emo_classifier=self.model_outputs_cache["loss_text_enc_emo_classifier"],
                    scores_disc_mp=scores_disc_mp,
                    feats_disc_mp=feats_disc_mp,
                    feats_disc_zp=feats_disc_zp,
                    pitch_loss=self.model_outputs_cache["pitch_loss"],
                    z_decoder_loss=self.model_outputs_cache["z_decoder_loss"],
                    conditional_module_loss=self.model_outputs_cache["conditional_module_loss"]
                )

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def _log(self, ap, batch, outputs, name_prefix="train"):  # pylint: disable=unused-argument,no-self-use
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]
        figures = plot_results(y_hat, y, ap, name_prefix)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name_prefix}/audio": sample_voice}

        alignments = outputs[1]["alignments"]
        align_img = alignments[0].data.cpu().numpy().T

        figures.update(
            {
                "alignment": plot_alignment(align_img, output_fig=False),
            }
        )
        return figures, audios

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ):  # pylint: disable=no-self-use
        """Create visualizations and waveform examples.

        For example, here you can plot spectrograms and generate sample sample waveforms from these spectrograms to
        be projected onto Tensorboard.

        Args:
            ap (AudioProcessor): audio processor used at training.
            batch (Dict): Model inputs used at the previous training step.
            outputs (Dict): Model outputs generated at the previoud training step.

        Returns:
            Tuple[Dict, np.ndarray]: training plots and output waveform.
        """
        figures, audios = self._log(self.ap, batch, outputs, "train")
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._log(self.ap, batch, outputs, "eval")
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def get_aux_input_from_test_sentences(self, sentence_info):
        if hasattr(self.config, "model_args"):
            config = self.config.model_args
        else:
            config = self.config

        # extract speaker and language info
        text, speaker_name, style_wav, language_name, emotion_name, style_speaker_name = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

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
                text, speaker_name, style_wav, language_name, emotion_name = sentence_info
            elif len(sentence_info) == 6:
                text, speaker_name, style_wav, language_name, emotion_name, style_speaker_name = sentence_info
        else:
            text = sentence_info

        if style_wav and style_speaker_name is None:
            raise RuntimeError(" [!] You must to provide the style_speaker_name for the style_wav !!")

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id, emotion_id, emotion_embedding, style_speaker_id, style_speaker_d_vector = (
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embeddings()
                else:
                    if speaker_name in self.speaker_manager.ids:
                        d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
                    else:
                        d_vector = self.speaker_manager.embeddings[speaker_name]["embedding"]

                    d_vector = np.array(d_vector)[None, :]  # [1 x embedding_dim]

                if style_wav is not None:
                    if style_speaker_name in self.speaker_manager.ids:
                        style_speaker_d_vector = self.speaker_manager.get_mean_embedding(
                            style_speaker_name, num_samples=None, randomize=False
                        )
                    else:
                        style_speaker_d_vector = self.speaker_manager.embeddings[style_speaker_name]["embedding"]

                    style_speaker_d_vector = np.array(style_speaker_d_vector)[None, :]

            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.ids[speaker_name]

                if style_wav is not None:
                    style_speaker_id = self.speaker_manager.ids[style_speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.ids[language_name]

        # get emotion id/embedding
        if hasattr(self, "emotion_manager"):
            if config.use_external_emotions_embeddings:
                if emotion_name is None:
                    emotion_embedding = self.emotion_manager.get_random_embeddings()
                else:
                    if emotion_name in self.emotion_manager.ids:
                        emotion_embedding = self.emotion_manager.get_mean_embedding(
                            emotion_name, num_samples=None, randomize=False
                        )
                    else:
                        emotion_embedding = self.emotion_manager.embeddings[emotion_name]["embedding"]

                    emotion_embedding = np.array(emotion_embedding)[None, :]

            elif config.use_emotion_embedding:
                if emotion_name is None:
                    emotion_id = self.emotion_manager.get_random_id()
                else:
                    emotion_id = self.emotion_manager.ids[emotion_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "style_speaker_id": style_speaker_id,
            "style_speaker_d_vector": style_speaker_d_vector,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
            "emotion_embedding": emotion_embedding,
            "emotion_ids": emotion_id,
        }

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
            wav, alignment, _, _ = synthesis(
                self,
                aux_inputs["text"],
                self.config,
                "cuda" in str(next(self.parameters()).device),
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                emotion_embedding=aux_inputs["emotion_embedding"],
                emotion_id=aux_inputs["emotion_ids"],
                style_speaker_id=aux_inputs["style_speaker_id"],
                style_speaker_d_vector=aux_inputs["style_speaker_d_vector"],
                use_griffin_lim=True,
                do_trim_silence=False,
            ).values()
            test_audios["{}-audio".format(idx)] = wav
            test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    def format_batch(self, batch: Dict) -> Dict:
        """Compute speaker, langugage IDs, d_vector and emotion embeddings for the batch if necessary."""
        speaker_ids = None
        language_ids = None
        d_vectors = None
        emotion_embeddings = None
        emotion_ids = None

        # get numerical speaker ids from speaker names
        if (
            self.speaker_manager is not None
            and self.speaker_manager.ids
            and (
                self.args.use_speaker_embedding
                or self.args.use_prosody_encoder
                or self.args.use_text_enc_spk_reversal_classifier
            )
        ):
            speaker_ids = [self.speaker_manager.ids[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            batch["speaker_ids"] = speaker_ids

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.embeddings and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.embeddings
            d_vectors = [d_vector_mapping[w]["embedding"] for w in batch["audio_files"]]
            d_vectors = torch.FloatTensor(d_vectors)

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.ids and self.args.use_language_embedding:
            language_ids = [self.language_manager.ids[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        # get emotion embedding
        if (
            self.emotion_manager is not None
            and self.emotion_manager.embeddings
            and self.args.use_external_emotions_embeddings
        ):
            emotion_mapping = self.emotion_manager.embeddings
            emotion_embeddings = [emotion_mapping[w]["embedding"] for w in batch["audio_files"]]
            emotion_embeddings = torch.FloatTensor(emotion_embeddings)

        if (
            self.emotion_manager is not None
            and self.emotion_manager.embeddings
            and (
                self.args.use_emotion_embedding
                or self.args.use_prosody_enc_emo_classifier
                or self.args.use_text_enc_emo_classifier
            )
        ):
            emotion_mapping = self.emotion_manager.embeddings
            emotion_names = [emotion_mapping[w]["name"] for w in batch["audio_files"]]
            emotion_ids = [self.emotion_manager.ids[en] for en in emotion_names]
            emotion_ids = torch.LongTensor(emotion_ids)

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
        batch["speaker_ids"] = speaker_ids
        batch["emotion_embeddings"] = emotion_embeddings
        batch["emotion_ids"] = emotion_ids
        return batch

    def format_batch_on_device(self, batch):
        """Compute spectrograms on the device."""
        ac = self.config.audio

        if self.args.encoder_sample_rate:
            wav = self.audio_resampler(batch["waveform"])
        else:
            wav = batch["waveform"]

        # compute spectrograms
        batch["spec"] = wav_to_spec(wav, ac.fft_size, ac.hop_length, ac.win_length, center=False)

        if self.args.encoder_sample_rate:
            # recompute spec with high sampling rate to the loss
            spec_mel = wav_to_spec(batch["waveform"], ac.fft_size, ac.hop_length, ac.win_length, center=False)
            # remove extra stft frames if needed
            if spec_mel.size(2) > int(batch["spec"].size(2) * self.interpolate_factor):
                spec_mel = spec_mel[:, :, : int(batch["spec"].size(2) * self.interpolate_factor)]
            else:
                batch["spec"] = batch["spec"][:, :, : int(spec_mel.size(2) / self.interpolate_factor)]
        else:
            spec_mel = batch["spec"]

        batch["mel"] = spec_to_mel(
            spec=spec_mel,
            n_fft=ac.fft_size,
            num_mels=ac.num_mels,
            sample_rate=ac.sample_rate,
            fmin=ac.mel_fmin,
            fmax=ac.mel_fmax,
        )

        if self.args.encoder_sample_rate:
            assert batch["spec"].shape[2] == int(
                batch["mel"].shape[2] / self.interpolate_factor
            ), f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"
        else:
            assert batch["spec"].shape[2] == batch["mel"].shape[2], f"{batch['spec'].shape[2]}, {batch['mel'].shape[2]}"

        # compute spectrogram frame lengths
        batch["spec_lens"] = (batch["spec"].shape[2] * batch["waveform_rel_lens"]).int()
        batch["mel_lens"] = (batch["mel"].shape[2] * batch["waveform_rel_lens"]).int()

        if self.args.encoder_sample_rate:
            assert (batch["spec_lens"] - (batch["mel_lens"] / self.interpolate_factor).int()).sum() == 0
        else:
            assert (batch["spec_lens"] - batch["mel_lens"]).sum() == 0

        # zero the padding frames
        batch["spec"] = batch["spec"] * sequence_mask(batch["spec_lens"]).unsqueeze(1)
        batch["mel"] = batch["mel"] * sequence_mask(batch["mel_lens"]).unsqueeze(1)
        return batch

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
            dataset = VitsDataset(
                model_args=self.args,
                config=config,
                samples=samples,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                verbose=verbose,
                tokenizer=self.tokenizer,
                start_by_longest=config.start_by_longest,
                compute_pitch=config.get("compute_pitch", False),
                f0_cache_path=config.get("f0_cache_path", None),
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            if self.args.use_pitch:
                self.args.pitch_mean = dataset.f0_dataset.mean
                self.args.pitch_std = dataset.f0_dataset.std

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
                pin_memory=False,
            )
        return loader

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
        # select generator parameters
        optimizer0 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)

        gen_parameters = chain(params for k, params in self.named_parameters() if not k.startswith("disc."))
        optimizer1 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        return [optimizer0, optimizer1]

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
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsDiscriminatorLoss(self.config), VitsGeneratorLoss(self.config)]

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        eval=False,
        strict=True,
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}

        if self.args.encoder_sample_rate is not None and eval:
            # audio resampler is not used in inference time
            self.audio_resampler = None

        # handle fine-tuning from a checkpoint with additional speakers
        if hasattr(self, "emb_g") and state["model"]["emb_g.weight"].shape != self.emb_g.weight.shape:
            num_new_speakers = self.emb_g.weight.shape[0] - state["model"]["emb_g.weight"].shape[0]
            print(f" > Loading checkpoint with {num_new_speakers} additional speakers.")
            emb_g = state["model"]["emb_g.weight"]
            new_row = torch.randn(num_new_speakers, emb_g.shape[1])
            emb_g = torch.cat([emb_g, new_row], axis=0)
            state["model"]["emb_g.weight"] = emb_g
        # load the model weights
        self.load_state_dict(state["model"], strict=strict)

        if eval:
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "VitsConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        if not config.model_args.encoder_sample_rate:
            assert (
                upsample_rate == config.audio.hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {config.audio.hop_length}"
        else:
            encoder_to_vocoder_upsampling_factor = config.audio.sample_rate / config.model_args.encoder_sample_rate
            effective_hop_length = config.audio.hop_length * encoder_to_vocoder_upsampling_factor
            assert (
                upsample_rate == effective_hop_length
            ), f" [!] Product of upsample rates must be equal to the hop length - {upsample_rate} vs {effective_hop_length}"

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        language_manager = LanguageManager.init_from_config(config)
        emotion_manager = EmotionManager.init_from_config(config)

        if config.model_args.encoder_model_path and speaker_manager is not None:
            speaker_manager.init_encoder(config.model_args.encoder_model_path, config.model_args.encoder_config_path)
        if config.model_args.encoder_model_path and emotion_manager is not None:
            emotion_manager.init_encoder(config.model_args.encoder_model_path, config.model_args.encoder_config_path)
        return Vits(new_config, ap, tokenizer, speaker_manager, language_manager, emotion_manager=emotion_manager)


##################################
# VITS CHARACTERS
##################################


class VitsCharacters(BaseCharacters):
    """Characters class for VITs model for compatibility with pre-trained models"""

    def __init__(
        self,
        graphemes: str = _characters,
        punctuations: str = _punctuations,
        pad: str = _pad,
        ipa_characters: str = _phonemes,
    ) -> None:
        if ipa_characters is not None:
            graphemes += ipa_characters
        super().__init__(graphemes, punctuations, pad, None, None, "<BLNK>", is_unique=False, is_sorted=True)

    def _create_vocab(self):
        self._vocab = [self._pad] + list(self._punctuations) + list(self._characters) + [self._blank]
        self._char_to_id = {char: idx for idx, char in enumerate(self.vocab)}
        # pylint: disable=unnecessary-comprehension
        self._id_to_char = {idx: char for idx, char in enumerate(self.vocab)}

    @staticmethod
    def init_from_config(config: Coqpit):
        if config.characters is not None:
            _pad = config.characters["pad"]
            _punctuations = config.characters["punctuations"]
            _letters = config.characters["characters"]
            _letters_ipa = config.characters["phonemes"]
            return (
                VitsCharacters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
                config,
            )
        characters = VitsCharacters()
        new_config = replace(config, characters=characters.to_config())
        return characters, new_config

    def to_config(self) -> "CharactersConfig":
        return CharactersConfig(
            characters=self._characters,
            punctuations=self._punctuations,
            pad=self._pad,
            eos=None,
            bos=None,
            blank=self._blank,
            is_unique=False,
            is_sorted=True,
        )