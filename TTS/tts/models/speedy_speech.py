import torch
from torch import nn

from TTS.tts.layers.feed_forward.decoder import Decoder
from TTS.tts.layers.feed_forward.duration_predictor import DurationPredictor
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor
from TTS.tts.layers.feed_forward.encoder import Encoder
from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.glow_tts.monotonic_align import generate_path
from TTS.tts.utils.data import sequence_mask


class SpeedySpeech(nn.Module):
    """Speedy Speech model
    https://arxiv.org/abs/2008.03802

    Encoder -> DurationPredictor -> Decoder

    This model is able to achieve a reasonable performance with only
    ~3M model parameters and convolutional layers.

    This model requires precomputed phoneme durations to train a duration predictor. At inference
    it only uses the duration predictor to compute durations and expand encoder outputs respectively.

    Args:
        num_chars (int): number of unique input to characters
        out_channels (int): number of output tensor channels. It is equal to the expected spectrogram size.
        hidden_channels (int): number of channels in all the model layers.
        positional_encoding (bool, optional): enable/disable Positional encoding on encoder outputs. Defaults to True.
        length_scale (int, optional): coefficient to set the speech speed. <1 slower, >1 faster. Defaults to 1.
        encoder_type (str, optional): set the encoder type. Defaults to 'residual_conv_bn'.
        encoder_params (dict, optional): set encoder parameters depending on 'encoder_type'. Defaults to { "kernel_size": 4, "dilations": 4 * [1, 2, 4] + [1], "num_conv_blocks": 2, "num_res_blocks": 13 }.
        decoder_type (str, optional): decoder type. Defaults to 'residual_conv_bn'.
        decoder_params (dict, optional): set decoder parameters depending on 'decoder_type'. Defaults to { "kernel_size": 4, "dilations": 4 * [1, 2, 4, 8] + [1], "num_conv_blocks": 2, "num_res_blocks": 17 }.
        num_speakers (int, optional): number of speakers for multi-speaker training. Defaults to 0.
        external_c (bool, optional): enable external speaker embeddings. Defaults to False.
        c_in_channels (int, optional): number of channels in speaker embedding vectors. Defaults to 0.
    """

    # pylint: disable=dangerous-default-value

    def __init__(
        self,
        num_chars,
        out_channels,
        hidden_channels,
        positional_encoding=True,
        length_scale=1,
        encoder_type="residual_conv_bn",
        encoder_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 13
        },
        decoder_type="residual_conv_bn",
        decoder_params={
            "kernel_size": 4,
            "dilations": 4 * [1, 2, 4, 8] + [1],
            "num_conv_blocks": 2,
            "num_res_blocks": 17,
        },
        num_speakers=0,
        external_c=False,
        c_in_channels=0,
    ):

        super().__init__()
        self.length_scale = float(length_scale) if isinstance(
            length_scale, int) else length_scale
        self.emb = nn.Embedding(num_chars, hidden_channels)
        self.encoder = Encoder(hidden_channels, hidden_channels, encoder_type,
                               encoder_params, c_in_channels)
        if positional_encoding:
            self.pos_encoder = PositionalEncoding(hidden_channels)
        self.decoder = Decoder(out_channels, hidden_channels, decoder_type,
                               decoder_params)
        self.duration_predictor = DurationPredictor(hidden_channels +
                                                    c_in_channels)

        if num_speakers > 1 and not external_c:
            # speaker embedding layer
            self.emb_g = nn.Embedding(num_speakers, c_in_channels)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

        if c_in_channels > 0 and c_in_channels != hidden_channels:
            self.proj_g = nn.Conv1d(c_in_channels, hidden_channels, 1)

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
        o_en_ex = torch.matmul(
            attn.squeeze(1).transpose(1, 2), en.transpose(1,
                                                          2)).transpose(1, 2)
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
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.shape[1]),
                                 1).to(x.dtype)

        # encoder pass
        o_en = self.encoder(x_emb, x_mask)

        # speaker conditioning for duration predictor
        if g is not None:
            o_en_dp = self._concat_speaker_embedding(o_en, g)
        else:
            o_en_dp = o_en
        return o_en, o_en_dp, x_mask, g

    def _forward_decoder(self, o_en, o_en_dp, dr, x_mask, y_lengths, g):
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None),
                                 1).to(o_en_dp.dtype)
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

    def forward(self,
                x,
                x_lengths,
                y_lengths,
                dr,
                cond_input={
                    'x_vectors': None,
                    'speaker_ids': None
                }):  # pylint: disable=unused-argument
        """
        TODO: speaker embedding for speaker_ids
        Shapes:
            x: [B, T_max]
            x_lengths: [B]
            y_lengths: [B]
            dr: [B, T_max]
            g: [B, C]
        """
        g = cond_input['x_vectors'] if 'x_vectors' in cond_input else None
        o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
        o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
        o_de, attn = self._forward_decoder(o_en,
                                           o_en_dp,
                                           dr,
                                           x_mask,
                                           y_lengths,
                                           g=g)
        outputs = {
            'model_outputs': o_de.transpose(1, 2),
            'durations_log': o_dr_log.squeeze(1),
            'alignments': attn
        }
        return outputs

    def inference(self,
                  x,
                  cond_input={
                      'x_vectors': None,
                      'speaker_ids': None
                  }):  # pylint: disable=unused-argument
        """
        Shapes:
            x: [B, T_max]
            x_lengths: [B]
            g: [B, C]
        """
        g = cond_input['x_vectors'] if 'x_vectors' in cond_input else None
        x_lengths = torch.tensor(x.shape[1:2]).to(x.device)
        # input sequence should be greated than the max convolution size
        inference_padding = 5
        if x.shape[1] < 13:
            inference_padding += 13 - x.shape[1]
        # pad input to prevent dropping the last word
        x = torch.nn.functional.pad(x,
                                    pad=(0, inference_padding),
                                    mode="constant",
                                    value=0)
        o_en, o_en_dp, x_mask, g = self._forward_encoder(x, x_lengths, g)
        # duration predictor pass
        o_dr_log = self.duration_predictor(o_en_dp.detach(), x_mask)
        o_dr = self.format_durations(o_dr_log, x_mask).squeeze(1)
        y_lengths = o_dr.sum(1)
        o_de, attn = self._forward_decoder(o_en,
                                           o_en_dp,
                                           o_dr,
                                           x_mask,
                                           y_lengths,
                                           g=g)
        outputs = {
            'model_outputs': o_de.transpose(1, 2),
            'alignments': attn,
            'durations_log': None
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        text_input = batch['text_input']
        text_lengths = batch['text_lengths']
        mel_input = batch['mel_input']
        mel_lengths = batch['mel_lengths']
        x_vectors = batch['x_vectors']
        speaker_ids = batch['speaker_ids']
        durations = batch['durations']

        cond_input = {'x_vectors': x_vectors, 'speaker_ids': speaker_ids}
        outputs = self.forward(text_input, text_lengths, mel_lengths,
                               durations, cond_input)

        # compute loss
        loss_dict = criterion(outputs['model_outputs'], mel_input,
                              mel_lengths, outputs['durations_log'],
                              torch.log(1 + durations), text_lengths)

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(outputs['alignments'],
                                                   binary=True)
        loss_dict["align_error"] = align_error
        return outputs, loss_dict

    def train_log(self, ap: AudioProcessor, batch: dict, outputs: dict):
        model_outputs = outputs['model_outputs']
        alignments = outputs['alignments']
        mel_input = batch['mel_input']

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
        return figures, train_audio

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, ap: AudioProcessor, batch: dict, outputs: dict):
        return self.train_log(ap, batch, outputs)

    def load_checkpoint(self, config, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            assert not self.training
