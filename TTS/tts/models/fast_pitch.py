from dataclasses import dataclass, field
from typing import Dict, Tuple

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.aligner import AlignmentNetwork
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.duration_predictor import DurationPredictor
from TTS.tts.layers.glow_tts.monotonic_align import generate_path, maximum_path
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.data import sequence_mask
from TTS.tts.utils.visual import plot_alignment, plot_pitch, plot_spectrogram
from TTS.utils.audio import AudioProcessor


@dataclass
class FastPitchArgs(Coqpit):
    """Fast Pitch Model arguments.

    Args:

        num_chars (int):
            Number of characters in the vocabulary. Defaults to 100.

        out_channels (int):
            Number of output channels. Defaults to 80.

        hidden_channels (int):
            Number of base hidden channels of the model. Defaults to 512.

        num_speakers (int):
            Number of speakers for the speaker embedding layer. Defaults to 0.

        duration_predictor_hidden_channels (int):
            Number of hidden channels in the duration predictor. Defaults to 256.

        duration_predictor_dropout_p (float):
            Dropout rate for the duration predictor. Defaults to 0.1.

        duration_predictor_kernel_size (int):
            Kernel size of conv layers in the duration predictor. Defaults to 3.

        pitch_predictor_hidden_channels (int):
            Number of hidden channels in the pitch predictor. Defaults to 256.

        pitch_predictor_dropout_p (float):
            Dropout rate for the pitch predictor. Defaults to 0.1.

        pitch_predictor_kernel_size (int):
            Kernel size of conv layers in the pitch predictor. Defaults to 3.

        pitch_embedding_kernel_size (int):
            Kernel size of the projection layer in the pitch predictor. Defaults to 3.

        positional_encoding (bool):
            Whether to use positional encoding. Defaults to True.

        positional_encoding_use_scale (bool):
            Whether to use a learnable scale coeff in the positional encoding. Defaults to True.

        length_scale (int):
            Length scale that multiplies the predicted durations. Larger values result slower speech. Defaults to 1.0.

        encoder_type (str):
            Type of the encoder module. One of the encoders available in :class:`TTS.tts.layers.feed_forward.encoder`.
            Defaults to `fftransformer` as in the paper.

        encoder_params (dict):
            Parameters of the encoder module. Defaults to ```{"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}```

        decoder_type (str):
            Type of the decoder module. One of the decoders available in :class:`TTS.tts.layers.feed_forward.decoder`.
            Defaults to `fftransformer` as in the paper.

        decoder_params (str):
            Parameters of the decoder module. Defaults to ```{"hidden_channels_ffn": 1024, "num_heads": 1, "num_layers": 6, "dropout_p": 0.1}```

        use_d_vetor (bool):
            Whether to use precomputed d-vectors for multi-speaker training. Defaults to False.

        d_vector_dim (int):
            Number of channels of the d-vectors. Defaults to 0.

        detach_duration_predictor (bool):
            Detach the input to the duration predictor from the earlier computation graph so that the duraiton loss
            does not pass to the earlier layers. Defaults to True.

        max_duration (int):
            Maximum duration accepted by the model. Defaults to 75.

        use_aligner (bool):
            Use aligner network to learn the text to speech alignment. Defaults to True.
    """

    num_chars: int = None
    out_channels: int = 80
    hidden_channels: int = 384
    num_speakers: int = 0
    duration_predictor_hidden_channels: int = 256
    duration_predictor_kernel_size: int = 3
    duration_predictor_dropout_p: float = 0.1
    pitch_predictor_hidden_channels: int = 256
    pitch_predictor_kernel_size: int = 3
    pitch_predictor_dropout_p: float = 0.1
    pitch_embedding_kernel_size: int = 3
    positional_encoding: bool = True
    poisitonal_encoding_use_scale: bool = True
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
    detach_duration_predictor: bool = False
    max_duration: int = 75
    use_aligner: bool = True


class FastPitch(BaseTTS):
    """FastPitch model. Very similart to SpeedySpeech model but with pitch prediction.

    Paper::
        https://arxiv.org/abs/2006.06873

    Paper abstract::
        We present FastPitch, a fully-parallel text-to-speech model based on FastSpeech, conditioned on fundamental
        frequency contours. The model predicts pitch contours during inference. By altering these predictions,
        the generated speech can be more expressive, better match the semantic of the utterance, and in the end
        more engaging to the listener. Uniformly increasing or decreasing pitch with FastPitch generates speech
        that resembles the voluntary modulation of voice. Conditioning on frequency contours improves the overall
        quality of synthesized speech, making it comparable to state-of-the-art. It does not introduce an overhead,
        and FastPitch retains the favorable, fully-parallel Transformer architecture, with over 900x real-time
        factor for mel-spectrogram synthesis of a typical utterance."

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

        # don't use isintance not to import recursively
        if config.__class__.__name__ == "FastPitchConfig":
            if "characters" in config:
                # loading from FasrPitchConfig
                _, self.config, num_chars = self.get_characters(config)
                config.model_args.num_chars = num_chars
                self.args = self.config.model_args
            else:
                # loading from FastPitchArgs
                self.config = config
                self.args = config.model_args
        elif isinstance(config, FastPitchArgs):
            self.args = config
            self.config = config
        else:
            raise ValueError("config must be either a VitsConfig or Vitsself.args")

        self.max_duration = self.args.max_duration
        self.use_aligner = self.args.use_aligner
        self.use_binary_alignment_loss = False

        self.length_scale = (
            float(self.args.length_scale) if isinstance(self.args.length_scale, int) else self.args.length_scale
        )

        self.emb = nn.Embedding(self.args.num_chars, self.args.hidden_channels)

        self.encoder = Encoder(
            self.args.hidden_channels,
            self.args.hidden_channels,
            self.args.encoder_type,
            self.args.encoder_params,
            self.args.d_vector_dim,
        )

        if self.args.positional_encoding:
            self.pos_encoder = PositionalEncoding(self.args.hidden_channels)

        self.decoder = Decoder(
            self.args.out_channels,
            self.args.hidden_channels,
            self.args.decoder_type,
            self.args.decoder_params,
        )

        self.duration_predictor = DurationPredictor(
            self.args.hidden_channels + self.args.d_vector_dim,
            self.args.duration_predictor_hidden_channels,
            self.args.duration_predictor_kernel_size,
            self.args.duration_predictor_dropout_p,
        )

        self.pitch_predictor = DurationPredictor(
            self.args.hidden_channels + self.args.d_vector_dim,
            self.args.pitch_predictor_hidden_channels,
            self.args.pitch_predictor_kernel_size,
            self.args.pitch_predictor_dropout_p,
        )

        self.pitch_emb = nn.Conv1d(
            1,
            self.args.hidden_channels,
            kernel_size=self.args.pitch_embedding_kernel_size,
            padding=int((self.args.pitch_embedding_kernel_size - 1) / 2),
        )

        if self.args.num_speakers > 1 and not self.args.use_d_vector:
            # speaker embedding layer
            self.emb_g = nn.Embedding(self.args.num_speakers, self.args.d_vector_dim)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        if self.args.d_vector_dim > 0 and self.args.d_vector_dim != self.args.hidden_channels:
            self.proj_g = nn.Conv1d(self.args.d_vector_dim, self.args.hidden_channels, 1)

        if self.args.use_aligner:
            self.aligner = AlignmentNetwork(
                in_query_channels=self.args.out_channels, in_key_channels=self.args.hidden_channels
            )

    @staticmethod
    def generate_attn(dr, x_mask, y_mask=None):
        """Generate an attention mask from the durations.

        Shapes
           - dr: :math:`(B, T_{en})`
           - x_mask: :math:`(B, T_{en})`
           - y_mask: :math:`(B, T_{de})`
        """
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

        Shapes
            - en: :math:`(B, D_{en}, T_{en})`
            - dr: :math:`(B, T_{en})`
            - x_mask: :math:`(B, T_{en})`
            - y_mask: :math:`(B, T_{de})`

        Examples:
            - encoder output: :math:`[a,b,c,d]`
            - durations: :math:`[1, 3, 2, 1]`

            - expanded: :math:`[a, b, b, b, c, c, d]`
            - attention map: :math:`[[0, 0, 0, 0, 0, 0, 1], [0, 0, 0, 0, 1, 1, 0], [0, 1, 1, 1, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0]]`
        """
        attn = self.generate_attn(dr, x_mask, y_mask)
        o_en_ex = torch.matmul(attn.squeeze(1).transpose(1, 2).to(en.dtype), en.transpose(1, 2)).transpose(1, 2)
        return o_en_ex, attn

    def format_durations(self, o_dr_log, x_mask):
        """Format predicted durations.
        1. Convert to linear scale from log scale
        2. Apply the length scale for speed adjustment
        3. Apply masking.
        4. Cast 0 durations to 1.
        5. Round the duration values.

        Args:
            o_dr_log: Log scale durations.
            x_mask: Input text mask.

        Shapes:
            - o_dr_log: :math:`(B, T_{de})`
            - x_mask: :math:`(B, T_{en})`
        """
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

    def _forward_encoder(
        self, x: torch.LongTensor, x_mask: torch.FloatTensor, g: torch.FloatTensor = None
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Encoding forward pass.

        1. Embed speaker IDs if multi-speaker mode.
        2. Embed character sequences.
        3. Run the encoder network.
        4. Concat speaker embedding to the encoder output for the duration predictor.

        Args:
            x (torch.LongTensor): Input sequence IDs.
            x_mask (torch.FloatTensor): Input squence mask.
            g (torch.FloatTensor, optional): Conditioning vectors. In general speaker embeddings. Defaults to None.

        Returns:
            Tuple[torch.tensor, torch.tensor, torch.tensor, torch.tensor, torch.tensor]:
                encoder output, encoder output for the duration predictor, input sequence mask, speaker embeddings,
                character embeddings

        Shapes:
            - x: :math:`(B, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - g: :math:`(B, C)`
        """
        if hasattr(self, "emb_g"):
            g = nn.functional.normalize(self.emb_g(g))  # [B, C, 1]
        if g is not None:
            g = g.unsqueeze(-1)
        # [B, T, C]
        x_emb = self.emb(x)
        # encoder pass
        o_en = self.encoder(torch.transpose(x_emb, 1, -1), x_mask)
        # speaker conditioning for duration predictor
        if g is not None:
            o_en_dp = self._concat_speaker_embedding(o_en, g)
        else:
            o_en_dp = o_en
        return o_en, o_en_dp, x_mask, g, x_emb

    def _forward_decoder(
        self,
        o_en: torch.FloatTensor,
        dr: torch.IntTensor,
        x_mask: torch.FloatTensor,
        y_lengths: torch.IntTensor,
        g: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor]:
        """Decoding forward pass.

        1. Compute the decoder output mask
        2. Expand encoder output with the durations.
        3. Apply position encoding.
        4. Add speaker embeddings if multi-speaker mode.
        5. Run the decoder.

        Args:
            o_en (torch.FloatTensor): Encoder output.
            dr (torch.IntTensor): Ground truth durations or alignment network durations.
            x_mask (torch.IntTensor): Input sequence mask.
            y_lengths (torch.IntTensor): Output sequence lengths.
            g (torch.FloatTensor): Conditioning vectors. In general speaker embeddings.

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Decoder output, attention map from durations.
        """
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(o_en.dtype)
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
        return o_de.transpose(1, 2), attn.transpose(1, 2)

    def _forward_pitch_predictor(
        self,
        o_en: torch.FloatTensor,
        x_mask: torch.IntTensor,
        pitch: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
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

        Returns:
            Tuple[torch.FloatTensor, torch.FloatTensor]: Pitch embedding, pitch prediction.

        Shapes:
            - o_en: :math:`(B, C, T_{en})`
            - x_mask: :math:`(B, 1, T_{en})`
            - pitch: :math:`(B, 1, T_{de})`
            - dr: :math:`(B, T_{en})`
        """
        o_pitch = self.pitch_predictor(o_en, x_mask)
        if pitch is not None:
            avg_pitch = average_pitch(pitch, dr)
            o_pitch_emb = self.pitch_emb(avg_pitch)
            return o_pitch_emb, o_pitch, avg_pitch
        o_pitch_emb = self.pitch_emb(o_pitch)
        return o_pitch_emb, o_pitch

    def _forward_aligner(
        self, x: torch.FloatTensor, y: torch.FloatTensor, x_mask: torch.IntTensor, y_mask: torch.IntTensor
    ) -> Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """Aligner forward pass.

        1. Compute a mask to apply to the attention map.
        2. Run the alignment network.
        3. Apply MAS to compute the hard alignment map.
        4. Compute the durations from the hard alignment map.

        Args:
            x (torch.FloatTensor): Input sequence.
            y (torch.FloatTensor): Output sequence.
            x_mask (torch.IntTensor): Input sequence mask.
            y_mask (torch.IntTensor): Output sequence mask.

        Returns:
            Tuple[torch.IntTensor, torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
                Durations from the hard alignment map, soft alignment potentials, log scale alignment potentials,
                hard alignment map.

        Shapes:
            - x: :math:`[B, T_en, C_en]`
            - y: :math:`[B, T_de, C_de]`
            - x_mask: :math:`[B, 1, T_en]`
            - y_mask: :math:`[B, 1, T_de]`

            - o_alignment_dur: :math:`[B, T_en]`
            - alignment_soft: :math:`[B, T_en, T_de]`
            - alignment_logprob: :math:`[B, 1, T_de, T_en]`
            - alignment_mas: :math:`[B, T_en, T_de]`
        """
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        alignment_soft, alignment_logprob = self.aligner(y.transpose(1, 2), x.transpose(1, 2), x_mask, None)
        alignment_mas = maximum_path(
            alignment_soft.squeeze(1).transpose(1, 2).contiguous(), attn_mask.squeeze(1).contiguous()
        )
        o_alignment_dur = torch.sum(alignment_mas, -1).int()
        alignment_soft = alignment_soft.squeeze(1).transpose(1, 2)
        return o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas

    def forward(
        self,
        x: torch.LongTensor,
        x_lengths: torch.LongTensor,
        y_lengths: torch.LongTensor,
        y: torch.FloatTensor = None,
        dr: torch.IntTensor = None,
        pitch: torch.FloatTensor = None,
        aux_input: Dict = {"d_vectors": 0, "speaker_ids": None},  # pylint: disable=unused-argument
    ) -> Dict:
        """Model's forward pass.

        Args:
            x (torch.LongTensor): Input character sequences.
            x_lengths (torch.LongTensor): Input sequence lengths.
            y_lengths (torch.LongTensor): Output sequnce lengths. Defaults to None.
            y (torch.FloatTensor): Spectrogram frames. Defaults to None.
            dr (torch.IntTensor): Character durations over the spectrogram frames. Defaults to None.
            pitch (torch.FloatTensor): Pitch values for each spectrogram frame. Defaults to None.
            aux_input (Dict): Auxiliary model inputs. Defaults to `{"d_vectors": 0, "speaker_ids": None}`.

        Shapes:
            - x: :math:`[B, T_max]`
            - x_lengths: :math:`[B]`
            - y_lengths: :math:`[B]`
            - y: :math:`[B, T_max2]`
            - dr: :math:`[B, T_max]`
            - g: :math:`[B, C]`
            - pitch: :math:`[B, 1, T]`
        """
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        # compute sequence masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(y.dtype)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(y.dtype)
        # encoder pass
        o_en, o_en_dp, x_mask, g, x_emb = self._forward_encoder(x, x_mask, g)
        # duration predictor pass
        if self.args.detach_duration_predictor:
            o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
        else:
            o_dr_log = self.duration_predictor(o_en_dp, x_mask)
        o_dr = torch.clamp(torch.exp(o_dr_log) - 1, 0, self.max_duration)
        # generate attn mask from predicted durations
        o_attn = self.generate_attn(o_dr.squeeze(1), x_mask)
        # aligner pass
        if self.use_aligner:
            o_alignment_dur, alignment_soft, alignment_logprob, alignment_mas = self._forward_aligner(
                x_emb, y, x_mask, y_mask
            )
            dr = o_alignment_dur
        # pitch predictor pass
        o_pitch_emb, o_pitch, avg_pitch = self._forward_pitch_predictor(o_en_dp, x_mask, pitch, dr)
        o_en = o_en + o_pitch_emb
        # decoder pass
        o_de, attn = self._forward_decoder(o_en, dr, x_mask, y_lengths, g=g)
        outputs = {
            "model_outputs": o_de,
            "durations_log": o_dr_log.squeeze(1),
            "durations": o_dr.squeeze(1),
            "attn_durations": o_attn,  # for visualization
            "pitch_avg": o_pitch,
            "pitch_avg_gt": avg_pitch,
            "alignments": attn,
            "alignment_soft": alignment_soft.transpose(1, 2),
            "alignment_mas": alignment_mas.transpose(1, 2),
            "o_alignment_dur": o_alignment_dur,
            "alignment_logprob": alignment_logprob,
            "x_mask": x_mask,
            "y_mask": y_mask,
        }
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"d_vectors": None, "speaker_ids": None}):  # pylint: disable=unused-argument
        """Model's inference pass.

        Args:
            x (torch.LongTensor): Input character sequence.
            aux_input (Dict): Auxiliary model inputs. Defaults to `{"d_vectors": None, "speaker_ids": None}`.

        Shapes:
            - x: [B, T_max]
            - x_lengths: [B]
            - g: [B, C]
        """
        g = aux_input["d_vectors"] if "d_vectors" in aux_input else None
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]), 1).to(x.dtype).float()
        # encoder pass
        o_en, o_en_dp, x_mask, g, _ = self._forward_encoder(x, x_mask, g)
        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en_dp, x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)
        # pitch predictor pass
        o_pitch_emb, o_pitch = self._forward_pitch_predictor(o_en_dp, x_mask)
        o_en = o_en + o_pitch_emb
        # decoder pass
        o_de, attn = self._forward_decoder(o_en, o_dr, x_mask, y_lengths, g=g)
        outputs = {
            "model_outputs": o_de,
            "alignments": attn,
            "pitch": o_pitch,
            "durations_log": o_dr_log,
        }
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

        # forward pass
        outputs = self.forward(
            text_input, text_lengths, mel_lengths, y=mel_input, dr=durations, pitch=pitch, aux_input=aux_input
        )
        # use aligner's output as the duration target
        if self.use_aligner:
            durations = outputs["o_alignment_dur"]
        # use float32 in AMP
        with autocast(enabled=False):
            # compute loss
            loss_dict = criterion(
                decoder_output=outputs["model_outputs"],
                decoder_target=mel_input,
                decoder_output_lens=mel_lengths,
                dur_output=outputs["durations_log"],
                dur_target=durations,
                pitch_output=outputs["pitch_avg"],
                pitch_target=outputs["pitch_avg_gt"],
                input_lens=text_lengths,
                alignment_logprob=outputs["alignment_logprob"],
                alignment_soft=outputs["alignment_soft"] if self.use_binary_alignment_loss else None,
                alignment_hard=outputs["alignment_mas"] if self.use_binary_alignment_loss else None,
            )
            # compute duration error
            durations_pred = outputs["durations"]
            duration_error = torch.abs(durations - durations_pred).sum() / text_lengths.sum()
            loss_dict["duration_error"] = duration_error

        return outputs, loss_dict

    def train_log(self, ap: AudioProcessor, batch: dict, outputs: dict):  # pylint: disable=no-self-use
        model_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        mel_input = batch["mel_input"]
        pitch = batch["pitch"]
        pitch_avg_expanded, _ = self.expand_encoder_outputs(
            outputs["pitch_avg"], outputs["durations"], outputs["x_mask"], outputs["y_mask"]
        )

        pred_spec = model_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()
        pitch = pitch[0, 0].data.cpu().numpy()

        # TODO: denormalize before plotting
        pitch = abs(pitch)
        pitch_avg_expanded = abs(pitch_avg_expanded[0, 0]).data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
            "pitch_ground_truth": plot_pitch(pitch, gt_spec, ap, output_fig=False),
            "pitch_avg_predicted": plot_pitch(pitch_avg_expanded, pred_spec, ap, output_fig=False),
        }

        # plot the attention mask computed from the predicted durations
        if "attn_durations" in outputs:
            alignments_hat = outputs["attn_durations"][0].data.cpu().numpy()
            figures["alignment_hat"] = plot_alignment(alignments_hat.T, output_fig=False)

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

    def on_train_step_start(self, trainer):
        """Enable binary alignment loss when needed"""
        if trainer.total_steps_done > self.config.binary_align_loss_start_step:
            self.use_binary_alignment_loss = True


def average_pitch(pitch, durs):
    """Compute the average pitch value for each input character based on the durations.

    Shapes:
        - pitch: :math:`[B, 1, T_de]`
        - durs: :math:`[B, T_en]`
    """

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
