from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from coqpit import Coqpit
from torch import nn

from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.glow_tts.monotonic_align import generate_path
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.data import sequence_mask
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor


@dataclass
class FastPitchArgs(Coqpit):
    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 256
    num_speakers: int = 0
    duration_predictor_hidden_channels: int = 256
    duration_predictor_dropout: float = 0.1
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout_p: float = 0.1
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_dropout: float = 0.1
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    positional_encoding: bool = True
    length_scale: int = 1
    encoder_type: str = "fftransformer"
    encoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )
    decoder_type: str = "fftransformer"
    decoder_params: dict = field(
        default_factory=lambda: {"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}
    )
    use_d_vector: bool = False
    d_vector_dim: int = 0


class FastPitch(BaseTTS):
    """FastPitch model. Very similart to SpeedySpeech model but with pitch prediction.

    Paper abstract:
        We present FastPitch, a fully-parallel text-to-speech model based on FastSpeech, conditioned on fundamental
        frequency contours. The model predicts pitch contours during inference. By altering these predictions,
        the generated speech can be more expressive, better match the semantic of the utterance, and in the end
        more engaging to the listener. Uniformly increasing or decreasing pitch with FastPitch generates speech
        that resembles the voluntary modulation of voice. Conditioning on frequency contours improves the overall
        quality of synthesized speech, making it comparable to state-of-the-art. It does not introduce an overhead,
        and FastPitch retains the favorable, fully-parallel Transformer architecture, with over 900x real-time
        factor for mel-spectrogram synthesis of a typical utterance."

    Notes:
        TODO

    Args:
        config (Coqpit): Model coqpit class.

    Examples:
        >>> from TTS.tts.models.fast_pitch import FastPitch, FastPitchArgs
        >>> config = FastPitchArgs()
        >>> model = FastPitch(config)
    """

    # pylint: disable=dangerous-default-value
    def __init__(self, config: Coqpit):

        super().__init__()

        _, self.config, num_chars = self.get_characters(config)
        config.model_args.num_chars = num_chars

        self.length_scale = (
            float(config.model_args.length_scale)
            if isinstance(config.model_args.length_scale, int)
            else config.model_args.length_scale
        )

        self.emb = nn.Embedding(config.model_args.num_chars, config.model_args.hidden_channels)

        self.encoder = Encoder(
            config.model_args.hidden_channels,
            config.model_args.hidden_channels,
            config.model_args.encoder_type,
            config.model_args.encoder_params,
            config.model_args.d_vector_dim,
        )

        if config.model_args.positional_encoding:
            self.pos_encoder = PositionalEncoding(config.model_args.hidden_channels)

        self.decoder = Decoder(
            config.model_args.out_channels,
            config.model_args.hidden_channels,
            config.model_args.decoder_type,
            config.model_args.decoder_params,
        )

        self.duration_predictor = DurationPredictor(
            config.model_args.hidden_channels + config.model_args.d_vector_dim,
            config.model_args.duration_predictor_hidden_channels,
            config.model_args.duration_predictor_kernel_size,
            config.model_args.duration_predictor_dropout_p,
        )

        self.pitch_predictor = DurationPredictor(
            config.model_args.hidden_channels + config.model_args.d_vector_dim,
            config.model_args.pitch_predictor_hidden_channels,
            config.model_args.pitch_predictor_kernel_size,
            config.model_args.pitch_predictor_dropout_p,
        )

        self.pitch_emb = nn.Conv1d(
            1,
            config.model_args.hidden_channels,
            kernel_size=config.model_args.pitch_embedding_kernel_size,
            padding=int((config.model_args.pitch_embedding_kernel_size - 1) / 2),
        )

        self.register_buffer("pitch_mean", torch.zeros(1))
        self.register_buffer("pitch_std", torch.zeros(1))

        if config.model_args.num_speakers > 1 and not config.model_args.use_d_vector:
            # speaker embedding layer
            self.emb_g = nn.Embedding(config.model_args.num_speakers, config.model_args.d_vector_dim)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        if config.model_args.d_vector_dim > 0 and config.model_args.d_vector_dim != config.model_args.hidden_channels:
            self.proj_g = nn.Conv1d(config.model_args.d_vector_dim, config.model_args.hidden_channels, 1)

    @staticmethod
    def expand_encoder_outputs(en, dr, x_mask, y_mask):
        """Generate attention alignment map from durations and
        expand encoder outputs

        Example:
            encoder output: [a,b,c,d]
            durations: [1, 3, 2, 1]

            expanded: [a, b, b, b, c, c, d]
            attention map: [[0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 1, 1, 0],
                            [0, 1, 1, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0]]
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        attn = generate_path(dr, attn_mask.squeeze(1)).to(en.dtype)
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
            g = nn.functional.normalize(self.emb_g(g))  # [B, C, 1]

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

    def _forward_pitch_predictor(self, o_en, x_mask, pitch=None, dr=None):
        o_pitch = self.pitch_predictor(o_en, x_mask)
        if pitch is not None:
            avg_pitch = average_pitch(pitch, dr)
            o_pitch_emb = self.pitch_emb(avg_pitch)
            return o_pitch_emb, o_pitch, avg_pitch
        o_pitch_emb = self.pitch_emb(o_pitch)
        return o_pitch_emb, o_pitch

    def forward(
        self, x, x_lengths, y_lengths, dr, pitch, aux_input={"d_vectors": None, "speaker_ids": None}
    ):  # pylint: disable=unused-argument
        """
        Shapes:
            x: :math:`[B, T_max]`
            x_lengths: :math:`[B]`
            y_lengths: :math:`[B]`
            dr: :math:`[B, T_max]`
            g: :math:`[B, C]`
            pitch: :math:`[B, 1, T]`
        """
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
        o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
        o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(o_en_dp, x_mask, pitch, dr)
        o_en = o_en + o_pitch_emb
        o_de, attn = self._forward_decoder(o_en, o_en_dp, dr, x_mask, y_lengths, g=g)
        outputs = {
            "model_outputs": o_de.transpose(1, 2),
            "durations_log": o_dr_log.squeeze(1),
            "pitch": o_pitch,
            "pitch_gt": avg_pitch,
            "alignments": attn,
        }
        return outputs

    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):  # pylint: disable=unused-argument
        """
        Shapes:
            x: [B, T_max]
            x_lengths: [B]
            g: [B, C]
        """
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        # input sequence should be greated than the max convolution size
        inference_padding = 5
        if x.shape[1] < 13:
            inference_padding += 13 - x.shape[1]
        # pad input to prevent dropping the last word
        x = torch.nn.functional.pad(x, pad=(0, inference_padding), mode="constant", value=0)
        o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        # pitch predictor pass
        o_pitch_emb, o_pitch = self._forward_pitch_predictor(o_en_dp, x_mask)
        # if pitch_transform is not None:
        #     if self.pitch_std[0] == 0.0:
        #         # XXX LJSpeech-1.1 defaults
        #         mean, std = 218.14, 67.24
        #     else:
        #         mean, std = self.pitch_mean[0], self.pitch_std[0]
        #     pitch_pred = pitch_transform(pitch_pred, enc_mask.sum(dim=(1,2)), mean, std)

        # if pitch_tgt is None:
        #     pitch_emb = self.pitch_emb(pitch_pred.unsqueeze(1)).transpose(1, 2)
        # else:
        #     pitch_emb = self.pitch_emb(pitch_tgt.unsqueeze(1)).transpose(1, 2)
        o_en = o_en + o_pitch_emb
        y_lengths = o_dr.sum(1)
        o_de, attn = self._forward_decoder(o_en, o_en_dp, o_dr, x_mask, y_lengths, g=g)
        outputs = {"model_outputs": o_de.transpose(1, 2), "alignments": attn, "pitch": o_pitch, "durations_log": None}
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        pitch = batch["pitch"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]
        durations = batch["durations"]

        aux_input = {"d_vectors": d_vectors, "speaker_ids": speaker_ids}
        outputs = self.forward(text_input, text_lengths, mel_lengths, durations, pitch, aux_input)

        # compute loss
        loss_dict = criterion(
            outputs["model_outputs"],
            mel_input,
            mel_lengths,
            outputs["durations_log"],
            torch.log(1 + durations),
            outputs["pitch"],
            outputs["pitch_gt"],
            text_lengths,
        )

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(outputs["alignments"], binary=True)
        loss_dict["align_error"] = align_error
        return outputs, loss_dict

    def train_log(self, ap: AudioProcessor, batch: dict, outputs: dict):  # pylint: disable=no-self-use
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

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, ap: AudioProcessor, batch: dict, outputs: dict):
        return self.train_log(ap, batch, outputs)

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training

    def get_criterion(self):
        from TTS.tts.layers.losses import FastPitchLoss  # pylint: disable=import-outside-toplevel

        return FastPitchLoss(self.config)


def average_pitch(pitch, durs):
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    pitch_nonzero_cums = torch.nn.functional.pad(torch.cumsum(pitch != 0.0, dim=2), (1, 0))
    pitch_cums = torch.nn.functional.pad(torch.cumsum(pitch, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = pitch.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    pitch_sums = (torch.gather(pitch_cums, 2, dce) - torch.gather(pitch_cums, 2, dcs)).float()
    pitch_nelems = (torch.gather(pitch_nonzero_cums, 2, dce) - torch.gather(pitch_nonzero_cums, 2, dcs)).float()

    pitch_avg = torch.where(pitch_nelems == 0.0, pitch_nelems, pitch_sums / pitch_nelems)
    return pitch_avg
