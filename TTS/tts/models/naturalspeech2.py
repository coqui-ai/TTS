import math
import os
from dataclasses import dataclass, field, replace
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
import torch.distributed as dist
import torchaudio
from coqpit import Coqpit
from librosa.filters import mel as librosa_mel_fn
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import WeightedRandomSampler
from trainer.torch import DistributedSampler, DistributedSamplerWrapper
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.shared_configs import CharactersConfig
from TTS.tts.datasets.dataset import TTSDataset, _parse_sample
from TTS.tts.layers.naturalspeech2.encoder import TransformerEncoder
from TTS.tts.layers.naturalspeech2.predictor import ConvBlockWithPrompting
from TTS.tts.layers.naturalspeech2.diffusion import Diffusion
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.characters import BaseCharacters, _characters, _pad, _phonemes, _punctuations
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment
from TTS.utils.io import load_fsspec
from TTS.utils.samplers import BucketBatchSampler
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
        return_complex=False,
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
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
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
        mel = librosa_mel_fn(sr=sample_rate, n_fft=n_fft, n_mels=num_mels, fmin=fmin, fmax=fmax)
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
        return_complex=False,
    )

    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    spec = torch.matmul(mel_basis[fmax_dtype_device], spec)
    spec = amp_to_db(spec)
    return spec


#############################
# CONFIGS
#############################


@dataclass
class NaturalSpeech2AudioConfig(Coqpit):
    fft_size: int = 1024
    sample_rate: int = 22050
    win_length: int = 1024
    hop_length: int = 256
    num_mels: int = 80
    mel_fmin: int = 0
    mel_fmax: int = None


##############################
# DATASET
##############################


def get_attribute_balancer_weights(items: list, attr_name: str, multi_dict: dict = None):
    """Create inverse frequency weights for balancing the dataset.
    Use `multi_dict` to scale relative weights."""
    attr_names_samples = np.array([item[attr_name] for item in items])
    unique_attr_names = np.unique(attr_names_samples).tolist()
    attr_idx = [unique_attr_names.index(l) for l in attr_names_samples]
    attr_count = np.array([len(np.where(attr_names_samples == l)[0]) for l in unique_attr_names])
    weight_attr = 1.0 / attr_count
    dataset_samples_weight = np.array([weight_attr[l] for l in attr_idx])
    dataset_samples_weight = dataset_samples_weight / np.linalg.norm(dataset_samples_weight)
    if multi_dict is not None:
        # check if all keys are in the multi_dict
        for k in multi_dict:
            assert k in unique_attr_names, f"{k} not in {unique_attr_names}"
        # scale weights
        multiplier_samples = np.array([multi_dict.get(item[attr_name], 1.0) for item in items])
        dataset_samples_weight *= multiplier_samples
    return (
        torch.from_numpy(dataset_samples_weight).float(),
        unique_attr_names,
        np.unique(dataset_samples_weight).tolist(),
    )


class NaturalSpeech2Dataset(TTSDataset):
    def __init__(self, model_args, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pad_id = self.tokenizer.characters.pad_id
        self.model_args = model_args

    def __getitem__(self, idx):
        item = self.samples[idx]
        raw_text = item["text"]

        wav, _ = load_audio(item["audio_file"])
        if self.model_args.encoder_sample_rate is not None:
            if wav.size(1) % self.model_args.encoder_sample_rate != 0:
                wav = wav[:, : -int(wav.size(1) % self.model_args.encoder_sample_rate)]

        wav_filename = os.path.basename(item["audio_file"])

        token_ids = self.get_token_ids(idx, item["text"])

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
            "audio_unique_name": item["audio_unique_name"],
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
            - audio_unique_names: :math:`[B]`
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

        return {
            "tokens": token_padded,
            "token_lens": token_lens,
            "token_rel_lens": token_rel_lens,
            "waveform": wav_padded,  # (B x T)
            "waveform_lens": wav_lens,  # (B)
            "waveform_rel_lens": wav_rel_lens,
            "speaker_names": batch["speaker_name"],
            "language_names": batch["language_name"],
            "audio_files": batch["wav_file"],
            "raw_text": batch["raw_text"],
            "audio_unique_names": batch["audio_unique_name"],
        }


##############################
# MODEL DEFINITION
##############################


@dataclass
class NaturalSpeech2Args(Coqpit):
    """NaturalSpeech2 model arguments.

    Args:
    """

    # DurationPredictor params
    dp_hidden_dim: int = 512
    dp_n_layers: int = 30
    dp_n_attentions: int = 10
    dp_attention_head: int = 8
    dp_dropout: float = 0.5

    # PitchPredictor params
    pp_hidden_dim: int = 512
    pp_n_layers: int = 30
    pp_n_attentions: int = 10
    pp_attention_head: int = 8
    pp_dropout: float = 0.2

    # PeomptEncoder params
    pre_hidden_dim = 512
    pre_nhead = 8
    pre_n_layers = 6
    pre_dim_feedforward = 2048
    pre_kernel_size = 9
    pre_dropout = 0.1

    # PhonemeEncoder params
    phe_hidden_dim = 512
    phe_nhead = 8
    phe_n_layers = 6
    phe_dim_feedforward = 2048
    phe_kernel_size = 9
    phe_dropout = 0.1

    #diffusion params
    max_step: int = 1000
    diff_size: int = 512
    pre_attention_query_token: int = 32
    pre_attention_query_size: int = 512
    pre_attention_head: int = 8
    wavenet_kernel_size: int = 3
    wavenet_dilation: int = 2
    wavenet_stack: int = 40
    wavenet_dropout_rate: float = 0.2
    wavenet_attention_apply_in_stack: int = 3
    wavenet_attention_head: int = 8
    num_cervq_sample: int = 4
    noise_schedule: str = "sigmoid"  # you might want to add this, wasn't clear from your code

class NaturalSpeech2(BaseTTS):
    """NaturalSpeech2 TTS model

    Paper::
        https://arxiv.org/pdf/2304.09116.pdf

    Paper Abstract::
        Scaling text-to-speech (TTS) to large-scale, multi-speaker, and in-the-wild datasets
        is important to capture the diversity in human speech such as speaker identities,
        prosodies, and styles (e.g., singing). Current large TTS systems usually quantize
        speech into discrete tokens and use language models to generate these tokens one
        by one, which suffer from unstable prosody, word skipping/repeating issue, and
        poor voice quality. In this paper, we develop NaturalSpeech 2, a TTS system
        that leverages a neural audio codec with residual vector quantizers to get the
        quantized latent vectors and uses a diffusion model to generate these latent vectors
        conditioned on text input. To enhance the zero-shot capability that is important
        to achieve diverse speech synthesis, we design a speech prompting mechanism to
        facilitate in-context learning in the diffusion model and the duration/pitch predictor.
        We scale NaturalSpeech 2 to large-scale datasets with 44K hours of speech and
        singing data and evaluate its voice quality on unseen speakers. NaturalSpeech 2
        outperforms previous TTS systems by a large margin in terms of prosody/timbre
        similarity, robustness, and voice quality in a zero-shot setting, and performs novel
        zero-shot singing synthesis with only a speech prompt. Audio samples are available
        at https://speechresearch.github.io/naturalspeech2.

    Check :class:`TTS.tts.configs.naturalspeech2_config.NaturalSpeech2Config` for class arguments.

    Examples:
        >>> from TTS.tts.configs.v import NaturalSpeech2Config
        >>> from TTS.tts.models.naturalspeech2 import NaturalSpeech2
        >>> config = NaturalSpeech2Config()
        >>> model = NaturalSpeech2(config)
    """

    def __init__(
        self,
        config: Coqpit,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager, language_manager)

        self.init_multilingual(config)

        self.spec_segment_size = self.args.spec_segment_size

        self.phoneme_encoder = TransformerEncoder(
            self.args.phe_hidden_dim,
            self.args.phe_nhead,
            self.args.phe_n_layers,
            self.args.phe_dim_feedforward,
            self.args.phe_kernel_size,
            self.args.phe_dropout,
            language_emb_dim=self.embedded_language_dim,
        )

        self.prompt_encoder = TransformerEncoder(
            self.args.pre_hidden_dim,
            self.args.pre_nhead,
            self.args.pre_n_layers,
            self.args.pre_dim_feedforward,
            self.args.pre_kernel_size,
            self.args.pre_dropout,
            language_emb_dim=self.embedded_language_dim,
        )

        self.duration_predictor = ConvBlockWithPrompting(
            self.args.dp_hidden_dim,
            self.args.dp_n_layers,
            self.args.dp_n_attentions,
            self.args.dp_attention_head,
            self.args.dp_dropout,
        )
        self.f0_embedding = nn.Conv1d(
            in_channels= 1,
            out_channels= self.args.diff_size,
            kernel_size= 1,
        )
        
        self.pitch_predictor = ConvBlockWithPrompting(
            self.args.pp_hidden_dim,
            self.args.pp_n_layers,
            self.args.pp_n_attentions,
            self.args.pp_attention_head,
            self.args.pp_dropout,
        )

        self.diffusion = Diffusion(
            max_step=self.args.max_step,
            size_=self.args.diff_size,
            pre_attention_query_token=self.args.pre_attention_query_token,
            pre_attention_query_size=self.args.pre_attention_query_size,
            pre_attention_head=self.args.pre_attention_head,
            wavenet_kernel_size=self.args.wavenet_kernel_size,
            wavenet_dilation=self.args.wavenet_dilation,
            wavenet_stack=self.args.wavenet_stack,
            wavenet_dropout_rate=self.args.wavenet_dropout_rate,
            wavenet_attention_apply_in_stack=self.args.wavenet_attention_apply_in_stack,
            wavenet_attention_head=self.args.wavenet_attention_head,
            noise_schedule=self.args.noise_schedule
        )
        
        self.aligner = AlignmentNetwork(
            in_query_channels=self.args.out_channels, in_key_channels=self.args.hidden_channels
        )

    @property
    def device(self):
        return next(self.parameters()).device

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

    def on_epoch_start(self, trainer):  # pylint: disable=W0613
        """Freeze layers at the beginning of an epoch"""
        self._freeze_layers()

    def _freeze_layers(self):
        if self.args.freeze_phoneme_encoder:
            for param in self.phoneme_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_prompt_encoder:
            for param in self.prompt_encoder.parameters():
                param.requires_grad = False

        if self.args.freeze_duration_predictor:
            for param in self.duration_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_pitch_predictor:
            for param in self.pitch_predictor.parameters():
                param.requires_grad = False

        if self.args.freeze_diffusion:
            for param in self.diffusion.parameters():
                param.requires_grad = False

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        # [TODO] use this  
        return None
    def expand_sequence(phoneme_hidden, duration_prediction):
        expanded_sequence = []
        for i, phoneme in enumerate(phoneme_hidden):
            repeat_count = int(duration_prediction[i].item())
            expanded_sequence.extend([phoneme] * repeat_count)
        return torch.stack(expanded_sequence)

    # add pitch to duraion after adding pitch to durations add this to diffusion model
    # [TODO] check if we need to add average over durations function
    def add_pitch_information(expanded_sequence, pitch_prediction):
        return expanded_sequence + pitch_prediction.unsqueeze(-1)
    
    def forward(  # pylint: disable=dangerous-default-value
        self,
        latents: torch.tensor,
        latents_lengths: torch.tensor,
        mel: torch.tensor,
        mel_lengths: torch.tensor,
        prompt: torch.tensor,
        aux_input={"pitch": None, "durations": None, "language_ids": None},
    ) -> Dict:
        outputs = {}
        phoneme_enc = self.phoneme_encoder(latents)
        
        prompt_enc = self.prompt_encoder(prompt)
        
        durations = self.duration_predictor(phoneme_enc, prompt_enc)
        pitch = self.pitch_predictor(phoneme_enc, prompt_enc)
        
        phoneme_with_durations = self.expand_sequence(phoneme_enc, durations)
        expanded_encodings = self.add_pitch_information(phoneme_with_durations, pitch)
        
        latents, _, _= self.diffusion(encodings = expanded_encodings,
                                      lengths = latents_lengths,
                                      speech_prompts = prompt_enc,
                                      latents=latents)
        outputs.update(
            {
                "latents": latents,
                "durations": durations,
                "pitch": pitch
            }
        )
       
        return outputs

    @torch.no_grad()
    def inference(
        self,
        x,
        prompt
    ):  # pylint: disable=dangerous-default-value
        outputs = {

        }
        return outputs
    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int) -> Tuple[Dict, Dict]:
        
        codec_lens = batch["codec_lens"]
        latents = self.encodec.quantizer.decode(batch["codec"])
        predictions = self.encodec.decoder(latents).squeeze(1)  # [Batch, Audio_t]
        if optimizer_idx == 0:
            tokens = batch["tokens"]
            token_lenghts = batch["token_lens"]
            language_ids = batch["language_ids"]
            waveform = batch["waveform"]
            style_wav = batch["style_wav"]
            codec = batch["codec"]
            # generator pass
            outputs = self.forward(
                tokens,
                token_lenghts,
                codec,
                codec_lens,
                waveform,
                style_wav,
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute loss
            return outputs, loss_dict

        if optimizer_idx == 1:
            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
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
        else:
            text = sentence_info

        # get speaker  id/d_vector
        speaker_id, d_vector, language_id = None, None, None
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_embedding()
                else:
                    d_vector = self.speaker_manager.get_mean_embedding(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_id()
                else:
                    speaker_id = self.speaker_manager.name_to_id[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.name_to_id[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "language_id": language_id,
            "language_name": language_name,
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
                style_wav=aux_inputs["style_wav"],
                language_id=aux_inputs["language_id"],
                use_griffin_lim=True,
                do_trim_silence=False,
                codec=None
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
        """Compute langugage IDs and codec for the batch if necessary."""
        language_ids = None

        # get language ids from language names
        if self.language_manager is not None and self.language_manager.name_to_id and self.args.use_language_embedding:
            language_ids = [self.language_manager.name_to_id[ln] for ln in batch["language_names"]]

        if language_ids is not None:
            language_ids = torch.LongTensor(language_ids)

        batch["language_ids"] = language_ids

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

    def get_sampler(self, config: Coqpit, dataset: TTSDataset, num_gpus=1, is_eval=False):
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

        # input_audio_lenghts = [os.path.getsize(x["audio_file"]) for x in data_items]

        if weights is not None:
            w_sampler = WeightedRandomSampler(weights, len(weights))
            batch_sampler = BucketBatchSampler(
                w_sampler,
                data=data_items,
                batch_size=config.eval_batch_size if is_eval else config.batch_size,
                sort_key=lambda x: os.path.getsize(x["audio_file"]),
                drop_last=True,
            )
        else:
            batch_sampler = None
        # sampler for DDP
        if batch_sampler is None:
            batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        else:  # If a sampler is already defined use this sampler and DDP sampler together
            batch_sampler = (
                DistributedSamplerWrapper(batch_sampler) if num_gpus > 1 else batch_sampler
            )  # TODO: check batch_sampler with multi-gpu
        return batch_sampler

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
            dataset = NaturalSpeech2Dataset(
                model_args=self.args,
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
            )

            # wait all the DDP process to be ready
            if num_gpus > 1:
                dist.barrier()

            # sort input sequences from short to long
            dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(config, dataset, num_gpus)
            if sampler is None:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,  # shuffle is done in the dataset.
                    collate_fn=dataset.collate_fn,
                    drop_last=False,  # setting this False might cause issues in AMP training.
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                if num_gpus > 1:
                    loader = DataLoader(
                        dataset,
                        sampler=sampler,
                        batch_size=config.eval_batch_size if is_eval else config.batch_size,
                        collate_fn=dataset.collate_fn,
                        num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                        pin_memory=False,
                    )
                else:
                    loader = DataLoader(
                        dataset,
                        batch_sampler=sampler,
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
        scheduler_D = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            Naturalspeech2Loss
        )

        return [Naturalspeech2Loss(self.config)]

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, strict=True, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        # load the model weights
        self.load_state_dict(state["model"], strict=strict)

        if eval:
            self.eval()
            assert not self.training

    @staticmethod
    def init_from_config(config: "NaturalSpeech2Config", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (NaturalSpeech2Config): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        upsample_rate = torch.prod(torch.as_tensor(config.model_args.upsample_rates_decoder)).item()

        ap = AudioProcessor.init_from_config(config, verbose=verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        language_manager = LanguageManager.init_from_config(config)

        return NaturalSpeech2(new_config, ap, tokenizer, language_manager)

##################################
# NaturalSpeech2 CHARACTERS
##################################


class NaturalSpeech2Characters(BaseCharacters):
    """Characters class for NaturalSpeech2 model for compatibility with pre-trained models"""

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
                NaturalSpeech2Characters(graphemes=_letters, ipa_characters=_letters_ipa, punctuations=_punctuations, pad=_pad),
                config,
            )
        characters = NaturalSpeech2Characters()
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
