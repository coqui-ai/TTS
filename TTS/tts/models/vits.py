import math
from dataclasses import dataclass, field
from itertools import chain
from typing import Dict, List, Tuple

import torch

import torchaudio
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F

from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.vits.discriminator import VitsDiscriminator
from TTS.tts.layers.vits.networks import PosteriorEncoder, ResidualCouplingBlocks, TextEncoder
from TTS.tts.layers.vits.stochastic_duration_predictor import StochasticDurationPredictor
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, rand_segments, segment, sequence_mask
from TTS.tts.utils.languages import LanguageManager
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.visual import plot_alignment
from TTS.utils.trainer_utils import get_optimizer, get_scheduler
from TTS.vocoder.models.hifigan_generator import HifiganGenerator
from TTS.vocoder.utils.generic_utils import plot_results


@dataclass
class VitsArgs(Coqpit):
    """VITS model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels. Defaults to 513.

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

        speaker_encoder_config_path (str):
            Path to the file speaker encoder config file, to use for SCL. Defaults to "".

        speaker_encoder_model_path (str):
            Path to the file speaker encoder checkpoint file, to use for SCL. Defaults to "".

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
    detach_dp_input: bool = True
    use_language_embedding: bool = False
    embedded_language_dim: int = 4
    num_languages: int = 0
    language_ids_file: str = None
    use_speaker_encoder_as_loss: bool = False
    speaker_encoder_config_path: str = ""
    speaker_encoder_model_path: str = ""
    freeze_encoder: bool = False
    freeze_DP: bool = False
    freeze_PE: bool = False
    freeze_flow_decoder: bool = False
    freeze_waveform_decoder: bool = False


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

    # pylint: disable=dangerous-default-value

    def __init__(
        self,
        config: Coqpit,
        speaker_manager: SpeakerManager = None,
        language_manager: LanguageManager = None,
    ):

        super().__init__(config)

        self.END2END = True
        self.speaker_manager = speaker_manager
        self.language_manager = language_manager
        if config.__class__.__name__ == "VitsConfig":
            # loading from VitsConfig
            if "num_chars" not in config:
                _, self.config, num_chars = self.get_characters(config)
                config.model_args.num_chars = num_chars
            else:
                self.config = config
                config.model_args.num_chars = config.num_chars
            args = self.config.model_args
        elif isinstance(config, VitsArgs):
            # loading from VitsArgs
            self.config = config
            args = config
        else:
            raise ValueError("config must be either a VitsConfig or VitsArgs")

        self.args = args

        self.init_multispeaker(config)
        self.init_multilingual(config)

        self.length_scale = args.length_scale
        self.noise_scale = args.noise_scale
        self.inference_noise_scale = args.inference_noise_scale
        self.inference_noise_scale_dp = args.inference_noise_scale_dp
        self.noise_scale_dp = args.noise_scale_dp
        self.max_inference_len = args.max_inference_len
        self.spec_segment_size = args.spec_segment_size

        self.text_encoder = TextEncoder(
            args.num_chars,
            args.hidden_channels,
            args.hidden_channels,
            args.hidden_channels_ffn_text_encoder,
            args.num_heads_text_encoder,
            args.num_layers_text_encoder,
            args.kernel_size_text_encoder,
            args.dropout_p_text_encoder,
            language_emb_dim=self.embedded_language_dim,
        )

        self.posterior_encoder = PosteriorEncoder(
            args.out_channels,
            args.hidden_channels,
            args.hidden_channels,
            kernel_size=args.kernel_size_posterior_encoder,
            dilation_rate=args.dilation_rate_posterior_encoder,
            num_layers=args.num_layers_posterior_encoder,
            cond_channels=self.embedded_speaker_dim,
        )

        self.flow = ResidualCouplingBlocks(
            args.hidden_channels,
            args.hidden_channels,
            kernel_size=args.kernel_size_flow,
            dilation_rate=args.dilation_rate_flow,
            num_layers=args.num_layers_flow,
            cond_channels=self.embedded_speaker_dim,
        )

        if args.use_sdp:
            self.duration_predictor = StochasticDurationPredictor(
                args.hidden_channels,
                192,
                3,
                args.dropout_p_duration_predictor,
                4,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )
        else:
            self.duration_predictor = DurationPredictor(
                args.hidden_channels,
                256,
                3,
                args.dropout_p_duration_predictor,
                cond_channels=self.embedded_speaker_dim,
                language_emb_dim=self.embedded_language_dim,
            )

        self.waveform_decoder = HifiganGenerator(
            args.hidden_channels,
            1,
            args.resblock_type_decoder,
            args.resblock_dilation_sizes_decoder,
            args.resblock_kernel_sizes_decoder,
            args.upsample_kernel_sizes_decoder,
            args.upsample_initial_channel_decoder,
            args.upsample_rates_decoder,
            inference_padding=0,
            cond_channels=self.embedded_speaker_dim,
            conv_pre_weight_norm=False,
            conv_post_weight_norm=False,
            conv_post_bias=False,
        )

        if args.init_discriminator:
            self.disc = VitsDiscriminator(use_spectral_norm=args.use_spectral_norm_disriminator)

    def init_multispeaker(self, config: Coqpit):
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        You must provide a `speaker_manager` at initialization to set up the multi-speaker modules.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        self.embedded_speaker_dim = 0
        self.num_speakers = self.args.num_speakers

        if self.speaker_manager:
            self.num_speakers = self.speaker_manager.num_speakers

        if self.args.use_speaker_embedding:
            self._init_speaker_embedding()

        if self.args.use_d_vector_file:
            self._init_d_vector()

        # TODO: make this a function
        if self.args.use_speaker_encoder_as_loss:
            if self.speaker_manager.speaker_encoder is None and (
                not config.speaker_encoder_model_path or not config.speaker_encoder_config_path
            ):
                raise RuntimeError(
                    " [!] To use the speaker consistency loss (SCL) you need to specify speaker_encoder_model_path and speaker_encoder_config_path !!"
                )

            self.speaker_manager.speaker_encoder.eval()
            print(" > External Speaker Encoder Loaded !!")

            if (
                hasattr(self.speaker_manager.speaker_encoder, "audio_config")
                and self.config.audio["sample_rate"] != self.speaker_manager.speaker_encoder.audio_config["sample_rate"]
            ):
                self.audio_transform = torchaudio.transforms.Resample(
                        orig_freq=self.audio_config["sample_rate"],
                        new_freq=self.speaker_manager.speaker_encoder.audio_config["sample_rate"],
                        )
            else:
                self.audio_transform = None

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
            self.emb_l = None

    @staticmethod
    def _set_cond_input(aux_input: Dict):
        """Set the speaker conditioning input based on the multi-speaker mode."""
        sid, g, lid = None, None, None
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

        return sid, g, lid

    def get_aux_input(self, aux_input: Dict):
        sid, g, lid = self._set_cond_input(aux_input)
        return {"speaker_ids": sid, "style_wav": None, "d_vectors": g, "language_ids": lid}

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
                    d_vector = self.speaker_manager.get_random_d_vector()
                else:
                    d_vector = self.speaker_manager.get_mean_d_vector(speaker_name, num_samples=1, randomize=False)
            elif config.use_speaker_embedding:
                if speaker_name is None:
                    speaker_id = self.speaker_manager.get_random_speaker_id()
                else:
                    speaker_id = self.speaker_manager.speaker_ids[speaker_name]

        # get language id
        if hasattr(self, "language_manager") and config.use_language_embedding and language_name is not None:
            language_id = self.language_manager.language_id_mapping[language_name]

        return {
            "text": text,
            "speaker_id": speaker_id,
            "style_wav": style_wav,
            "d_vector": d_vector,
            "language_id": language_id,
            "language_name": language_name,
        }

    def forward(
        self,
        x: torch.tensor,
        x_lengths: torch.tensor,
        y: torch.tensor,
        y_lengths: torch.tensor,
        waveform: torch.tensor,
        aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None},
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
            - waveform: :math:`[B, T_wav, 1]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
            - language_ids: :math:`[B]`
        """
        outputs = {}
        sid, g, lid = self._set_cond_input(aux_input)
        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        # posterior encoder
        z, m_q, logs_q, y_mask = self.posterior_encoder(y, y_lengths, g=g)

        # flow layers
        z_p = self.flow(z, y_mask, g=g)

        # find the alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        with torch.no_grad():
            o_scale = torch.exp(-2 * logs_p)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.einsum("klm, kln -> kmn", [o_scale, -0.5 * (z_p ** 2)])
            logp3 = torch.einsum("klm, kln -> kmn", [m_p * o_scale, z_p])
            logp4 = torch.sum(-0.5 * (m_p ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp2 + logp3 + logp1 + logp4
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

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

        # expand prior
        m_p = torch.einsum("klmn, kjm -> kjn", [attn, m_p])
        logs_p = torch.einsum("klmn, kjm -> kjn", [attn, logs_p])

        # select a random feature segment for the waveform decoder
        z_slice, slice_ids = rand_segments(z, y_lengths, self.spec_segment_size)
        o = self.waveform_decoder(z_slice, g=g)

        wav_seg = segment(
            waveform,
            slice_ids * self.config.audio.hop_length,
            self.args.spec_segment_size * self.config.audio.hop_length,
        )

        if self.args.use_speaker_encoder_as_loss and self.speaker_manager.speaker_encoder is not None:
            # concate generated and GT waveforms
            wavs_batch = torch.cat((wav_seg, o), dim=0)

            # resample audio to speaker encoder sample_rate
            # pylint: disable=W0105
            if self.audio_transform is not None:
                wavs_batch = self.audio_transform(wavs_batch)

            pred_embs = self.speaker_manager.speaker_encoder.forward(wavs_batch, l2_norm=True)

            # split generated and GT speaker embeddings
            gt_spk_emb, syn_spk_emb = torch.chunk(pred_embs, 2, dim=0)
        else:
            gt_spk_emb, syn_spk_emb = None, None

        outputs.update(
            {
                "model_outputs": o,
                "alignments": attn.squeeze(1),
                "z": z,
                "z_p": z_p,
                "m_p": m_p,
                "logs_p": logs_p,
                "m_q": m_q,
                "logs_q": logs_q,
                "waveform_seg": wav_seg,
                "gt_spk_emb": gt_spk_emb,
                "syn_spk_emb": syn_spk_emb,
            }
        )
        return outputs

    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None, "language_ids": None}):
        """
        Shapes:
            - x: :math:`[B, T_seq]`
            - d_vectors: :math:`[B, C, 1]`
            - speaker_ids: :math:`[B]`
        """
        sid, g, lid = self._set_cond_input(aux_input)
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)

        # speaker embedding
        if self.args.use_speaker_embedding and sid is not None:
            g = self.emb_g(sid).unsqueeze(-1)

        # language embedding
        lang_emb = None
        if self.args.use_language_embedding and lid is not None:
            lang_emb = self.emb_l(lid).unsqueeze(-1)

        x, m_p, logs_p, x_mask = self.text_encoder(x, x_lengths, lang_emb=lang_emb)

        if self.args.use_sdp:
            logw = self.duration_predictor(
                x, x_mask, g=g, reverse=True, noise_scale=self.inference_noise_scale_dp, lang_emb=lang_emb
            )
        else:
            logw = self.duration_predictor(x, x_mask, g=g, lang_emb=lang_emb)

        w = torch.exp(logw) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = sequence_mask(y_lengths, None).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1).transpose(1, 2))

        m_p = torch.matmul(attn.transpose(1, 2), m_p.transpose(1, 2)).transpose(1, 2)
        logs_p = torch.matmul(attn.transpose(1, 2), logs_p.transpose(1, 2)).transpose(1, 2)

        z_p = m_p + torch.randn_like(m_p) * torch.exp(logs_p) * self.inference_noise_scale
        z = self.flow(z_p, y_mask, g=g, reverse=True)
        o = self.waveform_decoder((z * y_mask)[:, :, : self.max_inference_len], g=g)

        outputs = {"model_outputs": o, "alignments": attn.squeeze(1), "z": z, "z_p": z_p, "m_p": m_p, "logs_p": logs_p}
        return outputs

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
        elif self.args.use_speaker_embedding and self.args.use_d_vector_file:
            g_src = F.normalize(speaker_cond_src).unsqueeze(-1)
            g_tgt = F.normalize(speaker_cond_tgt).unsqueeze(-1)
        else:
            raise RuntimeError(" [!] Voice conversion is only supported on multi-speaker models.")

        z, _, _, y_mask = self.posterior_encoder(y.transpose(1, 2), y_lengths, g=g_src)
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
        # pylint: disable=attribute-defined-outside-init
        if optimizer_idx not in [0, 1]:
            raise ValueError(" [!] Unexpected `optimizer_idx`.")

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

        if optimizer_idx == 0:
            text_input = batch["text_input"]
            text_lengths = batch["text_lengths"]
            mel_lengths = batch["mel_lengths"]
            linear_input = batch["linear_input"]
            d_vectors = batch["d_vectors"]
            speaker_ids = batch["speaker_ids"]
            language_ids = batch["language_ids"]
            waveform = batch["waveform"]

            # generator pass
            outputs = self.forward(
                text_input,
                text_lengths,
                linear_input.transpose(1, 2),
                mel_lengths,
                waveform.transpose(1, 2),
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids, "language_ids": language_ids},
            )

            # cache tensors for the discriminator
            self.y_disc_cache = None
            self.wav_seg_disc_cache = None
            self.y_disc_cache = outputs["model_outputs"]
            self.wav_seg_disc_cache = outputs["waveform_seg"]

            # compute discriminator scores and features
            outputs["scores_disc_fake"], outputs["feats_disc_fake"], _, outputs["feats_disc_real"] = self.disc(
                outputs["model_outputs"], outputs["waveform_seg"]
            )

            # compute losses
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    waveform_hat=outputs["model_outputs"].float(),
                    waveform=outputs["waveform_seg"].float(),
                    z_p=outputs["z_p"].float(),
                    logs_q=outputs["logs_q"].float(),
                    m_p=outputs["m_p"].float(),
                    logs_p=outputs["logs_p"].float(),
                    z_len=mel_lengths,
                    scores_disc_fake=outputs["scores_disc_fake"],
                    feats_disc_fake=outputs["feats_disc_fake"],
                    feats_disc_real=outputs["feats_disc_real"],
                    loss_duration=outputs["loss_duration"],
                    use_speaker_encoder_as_loss=self.args.use_speaker_encoder_as_loss,
                    gt_spk_emb=outputs["gt_spk_emb"],
                    syn_spk_emb=outputs["syn_spk_emb"],
                )

        elif optimizer_idx == 1:
            # discriminator pass
            outputs = {}

            # compute scores and features
            outputs["scores_disc_fake"], _, outputs["scores_disc_real"], _ = self.disc(
                self.y_disc_cache.detach(), self.wav_seg_disc_cache
            )

            # compute loss
            with autocast(enabled=False):  # use float32 for the criterion
                loss_dict = criterion[optimizer_idx](
                    outputs["scores_disc_real"],
                    outputs["scores_disc_fake"],
                )
        return outputs, loss_dict

    def _log(self, ap, batch, outputs, name_prefix="train"):  # pylint: disable=unused-argument,no-self-use
        y_hat = outputs[0]["model_outputs"]
        y = outputs[0]["waveform_seg"]
        figures = plot_results(y_hat, y, ap, name_prefix)
        sample_voice = y_hat[0].squeeze(0).detach().cpu().numpy()
        audios = {f"{name_prefix}/audio": sample_voice}

        alignments = outputs[0]["alignments"]
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
        ap = assets["audio_processor"]
        self._log(ap, batch, outputs, "train")

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module, optimizer_idx: int):
        return self.train_step(batch, criterion, optimizer_idx)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        ap = assets["audio_processor"]
        return self._log(ap, batch, outputs, "eval")

    @torch.no_grad()
    def test_run(self, ap) -> Tuple[Dict, Dict]:
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
            try:
                aux_inputs = self.get_aux_input_from_test_sentences(s_info)
                wav, alignment, _, _ = synthesis(
                    self,
                    aux_inputs["text"],
                    self.config,
                    "cuda" in str(next(self.parameters()).device),
                    ap,
                    speaker_id=aux_inputs["speaker_id"],
                    d_vector=aux_inputs["d_vector"],
                    style_wav=aux_inputs["style_wav"],
                    language_id=aux_inputs["language_id"],
                    language_name=aux_inputs["language_name"],
                    enable_eos_bos_chars=self.config.enable_eos_bos_chars,
                    use_griffin_lim=True,
                    do_trim_silence=False,
                ).values()
                test_audios["{}-audio".format(idx)] = wav
                test_figures["{}-alignment".format(idx)] = plot_alignment(alignment.T, output_fig=False)
            except:  # pylint: disable=bare-except
                print(" !! Error creating Test Sentence -", idx)
        return test_figures, test_audios

    def get_optimizer(self) -> List:
        """Initiate and return the GAN optimizers based on the config parameters.

        It returnes 2 optimizers in a list. First one is for the generator and the second one is for the discriminator.

        Returns:
            List: optimizers.
        """
        gen_parameters = chain(
            self.text_encoder.parameters(),
            self.posterior_encoder.parameters(),
            self.flow.parameters(),
            self.duration_predictor.parameters(),
            self.waveform_decoder.parameters(),
        )
        # add the speaker embedding layer
        if hasattr(self, "emb_g") and self.args.use_speaker_embedding and not self.args.use_d_vector_file:
            gen_parameters = chain(gen_parameters, self.emb_g.parameters())
        # add the language embedding layer
        if hasattr(self, "emb_l") and self.args.use_language_embedding:
            gen_parameters = chain(gen_parameters, self.emb_l.parameters())

        optimizer0 = get_optimizer(
            self.config.optimizer, self.config.optimizer_params, self.config.lr_gen, parameters=gen_parameters
        )
        optimizer1 = get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr_disc, self.disc)
        return [optimizer0, optimizer1]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr_gen, self.config.lr_disc]

    def get_scheduler(self, optimizer) -> List:
        """Set the schedulers for each optimizer.

        Args:
            optimizer (List[`torch.optim.Optimizer`]): List of optimizers.

        Returns:
            List: Schedulers, one for each optimizer.
        """
        scheduler0 = get_scheduler(self.config.lr_scheduler_gen, self.config.lr_scheduler_gen_params, optimizer[0])
        scheduler1 = get_scheduler(self.config.lr_scheduler_disc, self.config.lr_scheduler_disc_params, optimizer[1])
        return [scheduler0, scheduler1]

    def get_criterion(self):
        """Get criterions for each optimizer. The index in the output list matches the optimizer idx used in
        `train_step()`"""
        from TTS.tts.layers.losses import (  # pylint: disable=import-outside-toplevel
            VitsDiscriminatorLoss,
            VitsGeneratorLoss,
        )

        return [VitsGeneratorLoss(self.config), VitsDiscriminatorLoss(self.config)]

    @staticmethod
    def make_symbols(config):
        """Create a custom arrangement of symbols used by the model. The output list of symbols propagate along the
        whole training and inference steps."""
        _pad = config.characters["pad"]
        _punctuations = config.characters["punctuations"]
        _letters = config.characters["characters"]
        _letters_ipa = config.characters["phonemes"]
        symbols = [_pad] + list(_punctuations) + list(_letters)
        if config.use_phonemes:
            symbols += list(_letters_ipa)
        return symbols

    @staticmethod
    def get_characters(config: Coqpit):
        if config.characters is not None:
            symbols = Vits.make_symbols(config)
        else:
            from TTS.tts.utils.text.symbols import (  # pylint: disable=import-outside-toplevel
                parse_symbols,
                phonemes,
                symbols,
            )

            config.characters = parse_symbols()
            if config.use_phonemes:
                symbols = phonemes
        num_chars = len(symbols) + getattr(config, "add_blank", False)
        return symbols, config, num_chars

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        # compat band-aid for the pre-trained models to not use the encoder baked into the model
        # TODO: consider baking the speaker encoder into the model and call it from there.
        # as it is probably easier for model distribution.
        state["model"] = {k: v for k, v in state["model"].items() if "speaker_encoder" not in k}
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
