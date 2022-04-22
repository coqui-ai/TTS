import os
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple, Union

import numpy as np
import pyworld as pw
import torch
import torch.distributed as dist
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.utils.data import DataLoader
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.datasets.dataset import F0Dataset, TTSDataset, _parse_sample
from TTS.tts.layers.losses import ForwardTTSE2eLoss, VitsDiscriminatorLoss
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.models.base_tts import BaseTTSE2E
from TTS.tts.models.forward_tts import ForwardTTS, ForwardTTSArgs
from TTS.tts.models.vits import load_audio, wav_to_mel
from TTS.tts.utils.helpers import rand_segments, segment, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_avg_pitch, plot_spectrogram
from TTS.utils.audio.numpy_transforms import build_mel_basis, compute_f0
from TTS.utils.audio.numpy_transforms import db_to_amp as db_to_amp_numpy
from TTS.utils.audio.numpy_transforms import mel_to_wav as mel_to_wav_numpy
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


##############################
# DATASET
##############################


class ForwardTTSE2eF0Dataset(F0Dataset):
    """Override F0Dataset to avoid the AudioProcessor."""

    def __init__(
        self,
        audio_config: "AudioConfig",
        samples: Union[List[List], List[Dict]],
        verbose=False,
        cache_path: str = None,
        precompute_num_workers=0,
        normalize_f0=True,
    ):
        self.audio_config = audio_config
        super().__init__(
            samples=samples,
            ap=None,
            verbose=verbose,
            cache_path=cache_path,
            precompute_num_workers=precompute_num_workers,
            normalize_f0=normalize_f0,
        )

    @staticmethod
    def _compute_and_save_pitch(config, wav_file, pitch_file=None):
        wav, _ = load_audio(wav_file)
        f0 = compute_f0(
            x=wav.numpy()[0], sample_rate=config.sample_rate, hop_length=config.hop_length, pitch_fmax=config.pitch_fmax
        )
        # skip the last F0 value to align with the spectrogram
        if wav.shape[1] % config.hop_length != 0:
            f0 = f0[:-1]
        if pitch_file:
            np.save(pitch_file, f0)
        return f0

    def compute_or_load(self, wav_file):
        """
        compute pitch and return a numpy array of pitch values
        """
        pitch_file = self.create_pitch_file_path(wav_file, self.cache_path)
        if not os.path.exists(pitch_file):
            pitch = self._compute_and_save_pitch(self.audio_config, wav_file, pitch_file)
        else:
            pitch = np.load(pitch_file)
        return pitch.astype(np.float32)


class ForwardTTSE2eDataset(TTSDataset):
    def __init__(self, *args, **kwargs):
        # don't init the default F0Dataset in TTSDataset
        compute_f0 = kwargs.pop("compute_f0", False)
        kwargs["compute_f0"] = False

        self.audio_config = kwargs["audio_config"]
        del kwargs["audio_config"]

        super().__init__(*args, **kwargs)

        self.compute_f0 = compute_f0
        self.pad_id = self.tokenizer.characters.pad_id
        if self.compute_f0:
            self.f0_dataset = ForwardTTSE2eF0Dataset(
                audio_config=self.audio_config,
                samples=self.samples,
                cache_path=kwargs["f0_cache_path"],
                precompute_num_workers=kwargs["precompute_num_workers"],
            )

    def __getitem__(self, idx):
        item = self.samples[idx]
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

        return {
            "raw_text": raw_text,
            "token_ids": token_ids,
            "token_len": len(token_ids),
            "wav": wav,
            "pitch": f0,
            "wav_file": wav_filename,
            "speaker_name": item["speaker_name"],
            "language_name": item["language"],
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
            - pitch :math:`[B, T]`
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

        pitch_lens = [p.shape[0] for p in batch["pitch"]]
        pitch_lens = torch.LongTensor(pitch_lens)
        pitch_lens_max = torch.max(pitch_lens)

        token_padded = torch.LongTensor(B, max_text_len)
        wav_padded = torch.FloatTensor(B, 1, wav_lens_max)
        pitch_padded = torch.FloatTensor(B, 1, pitch_lens_max)

        token_padded = token_padded.zero_() + self.pad_id
        wav_padded = wav_padded.zero_() + self.pad_id
        pitch_padded = pitch_padded.zero_() + self.pad_id

        for i in range(len(ids_sorted_decreasing)):
            token_ids = batch["token_ids"][i]
            token_padded[i, : batch["token_len"][i]] = torch.LongTensor(token_ids)

            wav = batch["wav"][i]
            wav_padded[i, :, : wav.size(1)] = torch.FloatTensor(wav)

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
        }


##############################
# CONFIG DEFINITIONS
##############################


@dataclass
class ForwardTTSE2eAudio(Coqpit):
    sample_rate: int = 22050
    hop_length: int = 256
    win_length: int = 1024
    fft_size: int = 1024
    mel_fmin: float = 0.0
    mel_fmax: float = 8000
    num_mels: int = 80
    pitch_fmax: float = 640.0


@dataclass
class ForwardTTSE2eArgs(ForwardTTSArgs):
    # vocoder_config: BaseGANVocoderConfig = None
    num_chars: int = 100
    encoder_out_channels: int = 80
    spec_segment_size: int = 80
    # duration predictor
    detach_duration_predictor: bool = True
    duration_predictor_dropout_p: float = 0.1
    # pitch predictor
    pitch_predictor_dropout_p: float = 0.1
    # discriminator
    init_discriminator: bool = True
    use_spectral_norm_discriminator: bool = False
    # model parameters
    detach_vocoder_input: bool = False
    hidden_channels: int = 256
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {
            "hidden_channels_ffn": 1024,
            "num_heads": 2,
            "num_layers": 4,
            "dropout_p": 0.1,
            "kernel_size_fft": 9,
        }
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {
            "hidden_channels_ffn": 1024,
            "num_heads": 2,
            "num_layers": 4,
            "dropout_p": 0.1,
            "kernel_size_fft": 9,
        }
    )
    # generator
    resblock_type_decoder: str = "1"
    resblock_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [3, 7, 11])
    resblock_dilation_sizes_decoder: List[List[int]] = field(default_factory=lambda: [[1, 3, 5], [1, 3, 5], [1, 3, 5]])
    upsample_rates_decoder: List[int] = field(default_factory=lambda: [8, 8, 2, 2])
    upsample_initial_channel_decoder: int = 512
    upsample_kernel_sizes_decoder: List[int] = field(default_factory=lambda: [16, 16, 4, 4])
    # multi-speaker params
    use_speaker_embedding: bool = False
    num_speakers: int = 0
    speakers_file: str = None
    d_vector_file: str = None
    speaker_embedding_channels: int = 384
    use_d_vector_file: bool = False
    d_vector_dim: int = 0


##############################
# MODEL DEFINITION
##############################


class ForwardTTSE2e(BaseTTSE2E):
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
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config=config, tokenizer=tokenizer, speaker_manager=speaker_manager)
        self._set_model_args(config)

        self.init_multispeaker(config)

        self.encoder_model = ForwardTTS(config=self.args, ap=None, tokenizer=tokenizer, speaker_manager=speaker_manager)
        # self.vocoder_model = GAN(config=self.args.vocoder_config, ap=ap)
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
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        # use Vits Discriminator for limiting VRAM use
        if self.args.init_discriminator:
            self.disc = VitsDiscriminator(use_spectral_norm=self.args.use_spectral_norm_discriminator)

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

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

    def _init_speaker_embedding(self):
        # pylint: disable=attribute-defined-outside-init
        if self.num_speakers > 0:
            print(" > initialization of speaker-embedding layers.")
            self.embedded_speaker_dim = self.args.speaker_embedding_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)

    def _init_d_vector(self):
        # pylint: disable=attribute-defined-outside-init
        if hasattr(self, "emb_g"):
            raise ValueError("[!] Speaker embedding layer already initialized before d_vector settings.")
        self.embedded_speaker_dim = self.args.d_vector_dim

    def get_aux_input(self, *args, **kwargs) -> Dict:
        return self.encoder_model.get_aux_input(*args, **kwargs)

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        spec_lengths: torch.LongTensor,
        spec: torch.FloatTensor,
        waveform: torch.FloatTensor,
        dr: torch.IntTensor = None,
        pitch: torch.FloatTensor = None,
        aux_input: Dict = {"d_vectors": None, "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> Dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            x_lengths (torch.LongTensor): Input sequence lengths.
            spec_lengths (torch.LongTensor): Spectrogram sequnce lengths. Defaults to None.
            spec (torch.FloatTensor): Spectrogram frames. Only used when the alignment network is on. Defaults to None.
            waveform (torch.FloatTensor): Waveform. Defaults to None.
            dr (torch.IntTensor): Character durations over the spectrogram frames. Only used when the alignment network is off. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Only used when the pitch predictor is on. Defaults to None.
            aux_input (Dict): Auxiliary model inputs for multi-speaker training. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - spec_lengths: :math:`[B]`
            - spec: :math:`[B, T_max2]`
            - waveform: :math:`[B, C, T_max2]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T]`
        """
        encoder_outputs = self.encoder_model(
            x=x, x_lengths=x_lengths, y_lengths=spec_lengths, y=spec, dr=dr, pitch=pitch, aux_input=aux_input
        )
        o_en_ex = encoder_outputs["o_en_ex"].transpose(1, 2)  # [B, C_en, T_max2] -> [B, T_max2, C_en]
        o_en_ex_slices, slice_ids = rand_segments(
            x=o_en_ex.transpose(1, 2),
            x_lengths=spec_lengths,
            segment_size=self.args.spec_segment_size,
            let_short_samples=True,
            pad_short=True,
        )
        vocoder_output = self.waveform_decoder(
            x=o_en_ex_slices.detach() if self.args.detach_vocoder_input else o_en_ex_slices,
            g=encoder_outputs["g"],
        )
        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.args.spec_segment_size * self.config.audio.hop_length,
            pad_short=True,
        )
        model_outputs = {**encoder_outputs}
        model_outputs["encoder_outputs"] = encoder_outputs["model_outputs"]
        model_outputs["model_outputs"] = vocoder_output
        model_outputs["waveform_seg"] = wav_seg
        model_outputs["slice_ids"] = slice_ids
        return model_outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):
        encoder_outputs = self.encoder_model.inference(x=x, aux_input=aux_input, skip_decoder=True)
        o_en_ex = encoder_outputs["o_en_ex"]
        vocoder_output = self.waveform_decoder(x=o_en_ex, g=encoder_outputs["g"])
        model_outputs = {**encoder_outputs}
        model_outputs["model_outputs"] = vocoder_output
        return model_outputs

    @torch.no_grad()
    def inference_spec_decoder(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):
        encoder_outputs = self.encoder_model.inference(x=x, aux_input=aux_input, skip_decoder=False)
        model_outputs = {**encoder_outputs}
        return model_outputs

    def train_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        if optimizer_idx == 0:
            tokens = batch["text_input"]
            token_lenghts = batch["text_lengths"]
            spec = batch["mel_input"]
            spec_lens = batch["mel_lengths"]
            waveform = batch["waveform"]  # [B, T, C] -> [B, C, T]
            pitch = batch["pitch"]
            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]

            # generator pass
            outputs = self.forward(
                x=tokens,
                x_lengths=token_lenghts,
                spec_lengths=spec_lens,
                spec=spec,
                waveform=waveform,
                pitch=pitch,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # cache tensors for the generator pass
            self.model_outputs_cache = outputs  # pylint: disable=attribute-defined-outside-init

            # compute scores and features
            scores_d_fake, _, scores_d_real, _ = self.disc(outputs["model_outputs"].detach(), outputs["waveform_seg"])

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    scores_disc_fake=scores_d_fake,
                    scores_disc_real=scores_d_real,
                )
            return outputs, loss_dict

        if optimizer_idx == 1:
            mel = batch["mel_input"].transpose(1, 2)

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

            # compute discriminator scores and features
            scores_d_fake, feats_d_fake, _, feats_d_real = self.disc(
                self.model_outputs_cache["model_outputs"], self.model_outputs_cache["waveform_seg"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    decoder_output=self.model_outputs_cache["encoder_outputs"],
                    decoder_target=batch["mel_input"],
                    decoder_output_lens=batch["mel_lengths"],
                    dur_output=self.model_outputs_cache["durations_log"],
                    dur_target=self.model_outputs_cache["aligner_durations"],
                    pitch_output=self.model_outputs_cache["pitch_avg"] if self.args.use_pitch else None,
                    pitch_target=self.model_outputs_cache["pitch_avg_gt"] if self.args.use_pitch else None,
                    input_lens=batch["text_lengths"],
                    waveform=self.model_outputs_cache["waveform_seg"],
                    waveform_hat=self.model_outputs_cache["model_outputs"],
                    aligner_logprob=self.model_outputs_cache["aligner_logprob"],
                    aligner_hard=self.model_outputs_cache["aligner_mas"],
                    aligner_soft=self.model_outputs_cache["aligner_soft"],
                    binary_loss_weight=self.encoder_model.binary_loss_weight,
                    feats_fake=feats_d_fake,
                    feats_real=feats_d_real,
                    scores_fake=scores_d_fake,
                    spec_slice=mel_slice,
                    spec_slice_hat=mel_slice_hat,
                )

                # compute duration error for logging
                durations_pred = self.model_outputs_cache["durations"]
                durations_target = self.model_outputs_cache["aligner_durations"]
                duration_error = torch.abs(durations_target - durations_pred).sum() / batch["text_lengths"].sum()
                loss_dict["duration_error"] = duration_error

            return self.model_outputs_cache, loss_dict

        raise ValueError(" [!] Unexpected `optimizer_idx`.")

    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def _log(self, batch, outputs, name_prefix="train"):
        figures, audios = {}, {}

        # encoder outputs
        model_outputs = outputs[1]["encoder_outputs"]
        alignments = outputs[1]["alignments"]
        mel_input = batch["mel_input"]

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, None, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, None, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        # plot pitch figures
        if self.args.use_pitch:
            pitch_avg = abs(outputs[1]["pitch_avg_gt"][0, 0].data.cpu().numpy())
            pitch_avg_hat = abs(outputs[1]["pitch_avg"][0, 0].data.cpu().numpy())
            chars = self.tokenizer.decode(batch["text_input"][0].data.cpu().numpy())
            pitch_figures = {
                "pitch_ground_truth": plot_avg_pitch(pitch_avg, chars, output_fig=False),
                "pitch_avg_predicted": plot_avg_pitch(pitch_avg_hat, chars, output_fig=False),
            }
            figures.update(pitch_figures)

        # plot the attention mask computed from the predicted durations
        if "attn_durations" in outputs[1]:
            alignments_hat = outputs[1]["attn_durations"][0].data.cpu().numpy()
            figures["alignment_hat"] = plot_alignment(alignments_hat.T, output_fig=False)

        # Sample audio
        encoder_audio = mel_to_wav_numpy(
            mel=db_to_amp_numpy(x=pred_spec.T, gain=1, base=None), mel_basis=self.__mel_basis, **self.config.audio
        )
        audios[f"{name_prefix}/encoder_audio"] = encoder_audio

        # vocoder outputs
        y_hat = outputs[1]["model_outputs"]
        y = outputs[1]["waveform_seg"]

        vocoder_figures = plot_results(y_hat=y_hat, y=y, audio_config=self.config.audio, name_prefix=name_prefix)
        figures.update(vocoder_figures)

        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios[f"{name_prefix}/real_audio"] = sample_voice
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
        text, speaker_name, style_wav, language_name = None, None, None, None  # pylint: disable=unused-variable

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
        speaker_id, d_vector, language_id = None, None, None  # pylint: disable=unused-variable
        if hasattr(self, "speaker_manager"):
            if config.use_d_vector_file:
                if speaker_name is None:
                    d_vector = self.speaker_manager.get_random_d_vector()
                else:
                    d_vector = self.speaker_manager.get_mean_d_vector(speaker_name, num_samples=None, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_speaker_id()
                else:
                    speaker_id = self.speaker_manager.speaker_ids[speaker_name]

        # get language id
        # if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
        # language_id = self.language_manager.language_id_mapping[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": None,
            "language_name": None,
        }

    def synthesize(self, text: str, speaker_id, language_id, d_vector):
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

        # if language_id is not None:
        #     language_id = id_to_torch(language_id, cuda=is_cuda)

        text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=is_cuda)
        text_inputs = text_inputs.unsqueeze(0)

        # synthesize voice
        outputs = self.inference(text_inputs, aux_input={"d_vectors": d_vector, "speaker_ids": speaker_id})

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

    def synthesize_with_gl(self, text: str, speaker_id, language_id, d_vector):
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

        # if language_id is not None:
        #     language_id = id_to_torch(language_id, cuda=is_cuda)

        text_inputs = numpy_to_torch(text_inputs, torch.long, cuda=is_cuda)
        text_inputs = text_inputs.unsqueeze(0)

        # synthesize voice
        outputs = self.inference_spec_decoder(text_inputs, aux_input={"d_vectors": d_vector, "speaker_ids": speaker_id})

        # collect outputs
        wav = mel_to_wav_numpy(
            mel=outputs["model_outputs"].cpu().numpy()[0].T, mel_basis=self.__mel_basis, **self.config.audio
        )
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
            outputs = self.synthesize(
                aux_inputs["text"],
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                language_id=aux_inputs["language_id"],
            )
            outputs_gl = self.synthesize_with_gl(
                aux_inputs["text"],
                speaker_id=aux_inputs["speaker_id"],
                d_vector=aux_inputs["d_vector"],
                language_id=aux_inputs["language_id"],
            )
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

        # get numerical speaker ids from speaker names
        if self.speaker_manager is not None and self.speaker_manager.speaker_ids and self.args.use_speaker_embedding:
            speaker_ids = [self.speaker_manager.speaker_ids[sn] for sn in batch["speaker_names"]]

        if speaker_ids is not None:
            speaker_ids = torch.LongTensor(speaker_ids)
            batch["speaker_ids"] = speaker_ids

        # get d_vectors from audio file names
        if self.speaker_manager is not None and self.speaker_manager.d_vectors and self.args.use_d_vector_file:
            d_vector_mapping = self.speaker_manager.d_vectors
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

        batch["language_ids"] = language_ids
        batch["d_vectors"] = d_vectors
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

        assert (
            batch["pitch"].shape[2] == batch["mel_input"].shape[2]
        ), f"{batch['pitch'].shape[2]}, {batch['mel'].shape[2]}"
        batch["mel_lengths"] = (batch["mel_input"].shape[2] * batch["waveform_rel_lens"]).int()

        # zero the padding frames
        batch["mel_input"] = batch["mel_input"] * sequence_mask(batch["mel_lengths"]).unsqueeze(1)
        batch["mel_input"] = batch["mel_input"].transpose(1, 2)
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
            dataset = ForwardTTSE2eDataset(
                samples=samples,
                audio_config=self.config.audio,
                batch_group_size=0 if is_eval else config.batch_group_size * config.batch_size,
                min_text_len=config.min_text_len,
                max_text_len=config.max_text_len,
                min_audio_len=config.min_audio_len,
                max_audio_len=config.max_audio_len,
                phoneme_cache_path=config.phoneme_cache_path,
                precompute_num_workers=config.precompute_num_workers,
                compute_f0=config.compute_f0,
                f0_cache_path=config.f0_cache_path,
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

    def get_criterion(self):
        return [VitsDiscriminatorLoss(self.config), ForwardTTSE2eLoss(self.config)]

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.
        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.
        Returns:
            List: optimizers.
        """
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
        scheduler_D = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler_G = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler_D, scheduler_G]

    def on_train_step_start(self, trainer):
        """Schedule binary loss weight."""
        self.encoder_model.config.binary_loss_warmup_epochs = self.config.binary_loss_warmup_epochs
        self.encoder_model.on_train_step_start(trainer)

    def on_init_start(self, trainer: "Trainer"):
        self.__mel_basis = build_mel_basis(
            sample_rate=self.config.audio.sample_rate,
            fft_size=self.config.audio.fft_size,
            num_mels=self.config.audio.num_mels,
            mel_fmax=self.config.audio.mel_fmax,
            mel_fmin=self.config.audio.mel_fmin,
        )

    @staticmethod
    def init_from_config(config: "ForwardTTSConfig", samples: Union[List[List], List[Dict]] = None, verbose=False):
        """Initiate model from config

        Args:
            config (ForwardTTSE2eConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio.processor import AudioProcessor

        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        # language_manager = LanguageManager.init_from_config(config)
        return ForwardTTSE2e(config=new_config, tokenizer=tokenizer, speaker_manager=speaker_manager)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):
        """Load model from a checkpoint created by the 👟"""
        # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_state_dict(self):
        """Custom state dict of the model with all the necessary components for inference."""
        save_state = {
            "config": self.config.to_dict(),
            "args": self.args.to_dict(),
            "model": self.state_dict
        }

        if hasattr(self, "emb_g"):
            save_state["speaker_ids"] = self.speaker_manager.speaker_ids

        if self.args.use_d_vector_file:
            # TODO: implement saving of d_vectors
            ...
        return save_state

    def save(self, config, checkpoint_path):
        """Save model to a file."""
        save_state = self.get_state_dict(config, checkpoint_path)
        torch.save(save_state, checkpoint_path)
