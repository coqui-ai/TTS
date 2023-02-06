from dataclasses import dataclass, field
from typing import Dict, List, Union

import torch
from coqpit import Coqpit
from torch import nn

from TTS.tts.layers.align_tts.mdn import MDNBlock
from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.duration_predictor import DurationPredictor
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.io import load_fsspec


@dataclass
class AlignTTSArgs(Coqpit):
    """
    Args:
        num_chars (int):
            number of unique input to characters
        out_channels (int):
            number of output tensor channels. It is equal to the expected spectrogram size.
        hidden_channels (int):
            number of channels in all the model layers.
        hidden_channels_ffn (int):
            number of channels in transformer's conv layers.
        hidden_channels_dp (int):
            number of channels in duration predictor network.
        num_heads (int):
            number of attention heads in transformer networks.
        num_transformer_layers (int):
            number of layers in encoder and decoder transformer blocks.
        dropout_p (int):
            dropout rate in transformer layers.
        length_scale (int, optional):
            coefficient to set the speech speed. <1 slower, >1 faster. Defaults to 1.
        num_speakers (int, optional):
            number of speakers for multi-speaker training. Defaults to 0.
        external_c (bool, optional):
            enable external speaker embeddings. Defaults to False.
        c_in_channels (int, optional):
            number of channels in speaker embedding vectors. Defaults to 0.
    """

    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 256
    hidden_channels_dp: int = 256
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 2, "num_layers": 6, "dropout_p": 0.1}
    )
    length_scale: float = 1.0
    num_speakers: int = 0
    use_speaker_embedding: bool = False
    use_d_vector_file: bool = False
    d_vector_dim: int = 0


class AlignTTS(BaseTTS):
    """AlignTTS with modified duration predictor.
    https://arxiv.org/pdf/2003.01950.pdf

    Encoder -> DurationPredictor -> Decoder

    Check :class:`AlignTTSArgs` for the class arguments.

    Paper Abstract:
        Targeting at both high efficiency and performance, we propose AlignTTS to predict the
        mel-spectrum in parallel. AlignTTS is based on a Feed-Forward Transformer which generates mel-spectrum from a
        sequence of characters, and the duration of each character is determined by a duration predictor.Instead of
        adopting the attention mechanism in Transformer TTS to align text to mel-spectrum, the alignment loss is presented
        to consider all possible alignments in training by use of dynamic programming. Experiments on the LJSpeech dataset s
        how that our model achieves not only state-of-the-art performance which outperforms Transformer TTS by 0.03 in mean
        option score (MOS), but also a high efficiency which is more than 50 times faster than real-time.

    Note:
        Original model uses a separate character embedding layer for duration predictor. However, it causes the
        duration predictor to overfit and prevents learning higher level interactions among characters. Therefore,
        we predict durations based on encoder outputs which has higher level information about input characters. This
        enables training without phases as in the original paper.

        Original model uses Transormers in encoder and decoder layers. However, here you can set the architecture
        differently based on your requirements using ```encoder_type``` and ```decoder_type``` parameters.

    Examples:
        >>> from TTS.tts.configs.align_tts_config import AlignTTSConfig
        >>> config = AlignTTSConfig()
        >>> model = AlignTTS(config)

    """

    # pylint: disable=dangerous-default-value

    def __init__(
        self,
        config: "AlignTTSConfig",
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)
        self.speaker_manager = speaker_manager
        self.phase = -1
        self.length_scale = (
            float(config.model_args.length_scale)
            if isinstance(config.model_args.length_scale, int)
            else config.model_args.length_scale
        )

        self.emb = nn.Embedding(self.config.model_args.num_chars, self.config.model_args.hidden_channels)

        self.embedded_speaker_dim = 0
        self.init_multispeaker(config)

        self.pos_encoder = PositionalEncoding(config.model_args.hidden_channels)
        self.encoder = Encoder(
            config.model_args.hidden_channels,
            config.model_args.hidden_channels,
            config.model_args.encoder_type,
            config.model_args.encoder_params,
            self.embedded_speaker_dim,
        )
        self.decoder = Decoder(
            config.model_args.out_channels,
            config.model_args.hidden_channels,
            config.model_args.decoder_type,
            config.model_args.decoder_params,
        )
        self.duration_predictor = DurationPredictor(config.model_args.hidden_channels_dp)

        self.mod_layer = nn.Conv1d(config.model_args.hidden_channels, config.model_args.hidden_channels, 1)

        self.mdn_block = MDNBlock(config.model_args.hidden_channels, 2 * config.model_args.out_channels)

        if self.embedded_speaker_dim > 0 and self.embedded_speaker_dim != config.model_args.hidden_channels:
            self.proj_g = nn.Conv1d(self.embedded_speaker_dim, config.model_args.hidden_channels, 1)

    @staticmethod
    def compute_log_probs(mu, log_sigma, y):
        # pylint: disable=protected-access, c-extension-no-member
        y = y.transpose(1, 2).unsqueeze(1)  # [B, 1, T1, D]
        mu = mu.transpose(1, 2).unsqueeze(2)  # [B, T2, 1, D]
        log_sigma = log_sigma.transpose(1, 2).unsqueeze(2)  # [B, T2, 1, D]
        expanded_y, expanded_mu = torch.broadcast_tensors(y, mu)
        exponential = -0.5 * torch.mean(
            torch._C._nn.mse_loss(expanded_y, expanded_mu, 0) / torch.pow(log_sigma.exp(), 2), dim=-1
        )  # B, L, T
        logp = exponential - 0.5 * log_sigma.mean(dim=-1)
        return logp

    def compute_align_path(self, mu, log_sigma, y, x_mask, y_mask):
        # find the max alignment path
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        log_p = self.compute_log_probs(mu, log_sigma, y)
        # [B, T_en, T_dec]
        attn = maximum_path(log_p, attn_mask.squeeze(1)).unsqueeze(1)
        dr_mas = torch.sum(attn, -1)
        return dr_mas.squeeze(1), log_p

    @staticmethod
    def generate_attn(dr, x_mask, y_mask=None):
        # compute decode mask from the durations
        if y_mask is None:
            y_lengths = dr.sum(1).long()
            y_lengths[y_lengths < 1] = 1
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(dr.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(dr.dtype)
        return attn

    def expand_encoder_outputs(self, en, dr, x_mask, y_mask):
        """Generate attention alignment map from durations and
        expand encoder outputs

        Examples::
            - encoder output: [a,b,c,d]
            - durations: [1, 3, 2, 1]

            - expanded: [a, b, b, b, c, c, d]
            - attention map: [[0, 0, 0, 0, 0, 0, 1],
                             [0, 0, 0, 0, 1, 1, 0],
                             [0, 1, 1, 1, 0, 0, 0],
                             [1, 0, 0, 0, 0, 0, 0]]
        """
        attn = self.generate_attn(dr, x_mask, y_mask)
        o_en_ex = torch.matmul(attn.squeeze(1).transpose(1, 2), en.transpose(1, 2)).transpose(1, 2)
        return o_en_ex, attn

    def format_durations(self, o_dr_log, x_mask):
        o_dr = (torch.exp(o_dr_log) - 1) * x_mask * self.length_scale
        o_dr[o_dr < 1] = 1.0
        o_dr = torch.round(o_dr)
        return o_dr

    @staticmethod
    def _concat_speaker_embedding(o_en, g):
        g_exp = g.expand(-1, -1, o_en.size(-1))  # [B, C, T_en]
        o_en = torch.cat([o_en, g_exp], 1)
        return o_en

    def _sum_speaker_embedding(self, x, g):
        # project g to decoder dim.
        if hasattr(self, "proj_g"):
            g = self.proj_g(g)

        return x + g

    def _forward_encoder(self, x, x_lengths, g=None):
        if hasattr(self, "emb_g"):
            g = nn.functional.normalize(self.speaker_embedding(g))  # [B, C, 1]

        if g is not None:
            g = g.unsqueeze(-1)

        # [B, T, C]
        x_emb = self.emb(x)
        # [B, C, T]
        x_emb = torch.transpose(x_emb, 1, -1)

        # compute sequence masks
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype)

        # encoder pass
        o_en = self.encoder(x_emb, x_mask)

        # speaker conditioning for duration predictor
        if g is not None:
            o_en_dp = self._concat_speaker_embedding(o_en, g)
        else:
            o_en_dp = o_en
        return o_en, o_en_dp, x_mask, g

    def _forward_decoder(self, o_en, o_en_dp, dr, x_mask, y_lengths, g):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en_dp.dtype)
        # expand o_en with durations
        o_en_ex, attn = self.expand_encoder_outputs(o_en, dr, x_mask, y_mask)
        # positional encoding
        if hasattr(self, "pos_encoder"):
            o_en_ex = self.pos_encoder(o_en_ex, y_mask)
        # speaker embedding
        if g is not None:
            o_en_ex = self._sum_speaker_embedding(o_en_ex, g)
        # decoder pass
        o_de = self.decoder(o_en_ex, y_mask, g=g)
        return o_de, attn.transpose(1, 2)

    def _forward_mdn(self, o_en, y, y_lengths, x_mask):
        # MAS potentials and alignment
        mu, log_sigma = self.mdn_block(o_en)
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
        dr_mas, logp = self.compute_align_path(mu, log_sigma, y, x_mask, y_mask)
        return dr_mas, mu, log_sigma, logp

    def forward(
        self, x, x_lengths, y, y_lengths, aux_input={"d_vectors": None}, phase=None
    ):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - y_lengths: :math:`[B]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
        """
        y = y.transpose(1, 2)
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        o_de, o_dr_log, dr_mas_log, attn, mu, log_sigma, logp = None, None, None, None, None, None, None
        if phase == 0:
            # train encoder and MDN
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en_dp.dtype)
            attn = self.generate_attn(dr_mas, x_mask, y_mask)
        elif phase == 1:
            # train decoder
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            dr_mas, _, _, _ = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en.detach(), o_en_dp.detach(), dr_mas.detach(), x_mask, y_lengths, g=g)
        elif phase == 2:
            # train the whole except duration predictor
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en, o_en_dp, dr_mas, x_mask, y_lengths, g=g)
        elif phase == 3:
            # train duration predictor
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            o_dr_log = self.duration_predictor(x, x_mask)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en, o_en_dp, dr_mas, x_mask, y_lengths, g=g)
            o_dr_log = o_dr_log.squeeze(1)
        else:
            o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
            o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
            dr_mas, mu, log_sigma, logp = self._forward_mdn(o_en, y, y_lengths, x_mask)
            o_de, attn = self._forward_decoder(o_en, o_en_dp, dr_mas, x_mask, y_lengths, g=g)
            o_dr_log = o_dr_log.squeeze(1)
        dr_mas_log = torch.log(dr_mas + 1).squeeze(1)
        outputs = {
            "model_outputs": o_de.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dr_log,
            "durations_mas_log": dr_mas_log,
            "mu": mu,
            "log_sigma": log_sigma,
            "logp": logp,
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None}):  # pylint: disable=unused-argument
        """
        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - g: :math:`[B, C]`
        """
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        # pad input to prevent dropping the last word
        # x = torch.nn.functional.pad(x, pad=(0, 5), mode='constant', value=0)
        o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
        # o_dr_log = self.duration_predictor(x, x_mask)
        o_dr_log = self.duration_predictor(o_en_dp, x_mask)
        # duration predictor pass
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)
        o_de, attn = self._forward_decoder(o_en, o_en_dp, o_dr, x_mask, y_lengths, g=g)
        outputs = {"model_outputs": o_de.transpose(1, 2), "alignments": attn}
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]

        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input, self.phase)
        loss_dict = criterion(
            outputs["logp"],
            outputs["model_outputs"],
            mel_input,
            mel_lengths,
            outputs["durations_log"],
            outputs["durations_mas_log"],
            text_lengths,
            phase=self.phase,
        )

        return outputs, loss_dict

    def _create_logs(self, batch, outputs, ap):  # pylint: disable=no-self-use
        model_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        mel_input = batch["mel_input"]

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        # Sample audio
        train_audio = ap.inv_melspectrogram(pred_spec.T)
        return figures, {"audio": train_audio}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_criterion(self):
        from TTS.tts.layers.losses import AlignTTSLoss  # pylint: disable=import-outside-toplevel

        return AlignTTSLoss(self.config)

    @staticmethod
    def _set_phase(config, global_step):
        """Decide AlignTTS training phase"""
        if isinstance(config.phase_start_steps, list):
            vals = [i < global_step for i in config.phase_start_steps]
            if not True in vals:
                phase = 0
            else:
                phase = (
                    len(config.phase_start_steps)
                    - [i < global_step for i in config.phase_start_steps][::-1].index(True)
                    - 1
                )
        else:
            phase = None
        return phase

    def on_epoch_start(self, trainer):
        """Set AlignTTS training phase on epoch start."""
        self.phase = self._set_phase(trainer.config, trainer.total_steps_done)

    @staticmethod
    def init_from_config(config: "AlignTTSConfig", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (AlignTTSConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return AlignTTS(new_config, ap, tokenizer, speaker_manager)
