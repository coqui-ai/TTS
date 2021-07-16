import math

import torch
from torch import nn
from torch.nn import functional as F

from TTS.tts.configs import GlowTTSConfig
from TTS.tts.layers.glow_tts.decoder import Decoder
from TTS.tts.layers.glow_tts.encoder import Encoder
from TTS.tts.layers.glow_tts.monotonic_align import generate_path, maximum_path
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.data import sequence_mask
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import get_speaker_manager
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor


class GlowTTS(BaseTTS):
    """Glow TTS models from https://arxiv.org/abs/2005.11129

    Paper abstract:
        Recently, text-to-speech (TTS) models such as FastSpeech and ParaNet have been proposed to generate
        mel-spectrograms from text in parallel. Despite the advantage, the parallel TTS models cannot be trained
        without guidance from autoregressive TTS models as their external aligners. In this work, we propose Glow-TTS,
        a flow-based generative model for parallel TTS that does not require any external aligner. By combining the
        properties of flows and dynamic programming, the proposed model searches for the most probable monotonic
        alignment between text and the latent representation of speech on its own. We demonstrate that enforcing hard
        monotonic alignments enables robust TTS, which generalizes to long utterances, and employing generative flows
        enables fast, diverse, and controllable speech synthesis. Glow-TTS obtains an order-of-magnitude speed-up over
        the autoregressive model, Tacotron 2, at synthesis with comparable speech quality. We further show that our
        model can be easily extended to a multi-speaker setting.

    Check :class:`TTS.tts.configs.glow_tts_config.GlowTTSConfig` for class arguments.

    Examples:
        >>> from TTS.tts.configs import GlowTTSConfig
        >>> from TTS.tts.models.glow_tts import GlowTTS
        >>> config = GlowTTSConfig()
        >>> model = GlowTTS(config)

    """

    def __init__(self, config: GlowTTSConfig):

        super().__init__()

        # pass all config fields to `self`
        # for fewer code change
        self.config = config
        for key in config:
            setattr(self, key, config[key])

        _, self.config, self.num_chars = self.get_characters(config)
        self.decoder_output_dim = config.out_channels

        self.init_multispeaker(config)

        # if is a multispeaker and c_in_channels is 0, set to 256
        self.c_in_channels = 0
        if self.num_speakers > 1:
            if self.d_vector_dim:
                self.c_in_channels = self.d_vector_dim
            elif self.c_in_channels == 0 and not self.d_vector_dim:
                # TODO: make this adjustable
                self.c_in_channels = 256

        self.encoder = Encoder(
            self.num_chars,
            out_channels=self.out_channels,
            hidden_channels=self.hidden_channels_enc,
            hidden_channels_dp=self.hidden_channels_dp,
            encoder_type=self.encoder_type,
            encoder_params=self.encoder_params,
            mean_only=self.mean_only,
            use_prenet=self.use_encoder_prenet,
            dropout_p_dp=self.dropout_p_dp,
            c_in_channels=self.c_in_channels,
        )

        self.decoder = Decoder(
            self.out_channels,
            self.hidden_channels_dec,
            self.kernel_size_dec,
            self.dilation_rate,
            self.num_flow_blocks_dec,
            self.num_block_layers,
            dropout_p=self.dropout_p_dec,
            num_splits=self.num_splits,
            num_squeeze=self.num_squeeze,
            sigmoid_scale=self.sigmoid_scale,
            c_in_channels=self.c_in_channels,
        )

    def init_multispeaker(self, config: "Coqpit", data: list = None) -> None:
        """Initialize multi-speaker modules of a model. A model can be trained either with a speaker embedding layer
        or with external `d_vectors` computed from a speaker encoder model.

        If you need a different behaviour, override this function for your model.

        Args:
            config (Coqpit): Model configuration.
            data (List, optional): Dataset items to infer number of speakers. Defaults to None.
        """
        # init speaker manager
        self.speaker_manager = get_speaker_manager(config, data=data)
        self.num_speakers = self.speaker_manager.num_speakers
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            self.embedded_speaker_dim = self.c_in_channels
            self.emb_g = nn.Embedding(self.num_speakers, self.embedded_speaker_dim)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)

    @staticmethod
    def compute_outputs(attn, o_mean, o_log_scale, x_mask):
        """Compute and format the mode outputs with the given alignment map"""
        y_mean = torch.matmul(attn.squeeze(1).transpose(1, 2), o_mean.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        y_log_scale = torch.matmul(attn.squeeze(1).transpose(1, 2), o_log_scale.transpose(1, 2)).transpose(
            1, 2
        )  # [b, t', t], [b, t, d] -> [b, d, t']
        # compute total duration with adjustment
        o_attn_dur = torch.log(1 + torch.sum(attn, -1)) * x_mask
        return y_mean, y_log_scale, o_attn_dur

    def forward(
        self, x, x_lengths, y, y_lengths=None, aux_input={"d_vectors": None}
    ):  # pylint: disable=dangerous-default-value
        """
        Shapes:
            - x: :math:`[B, T]`
            - x_lenghts::math:` B`
            - y: :math:`[B, T, C]`
            - y_lengths::math:` B`
            - g: :math:`[B, C] or B`
        """
        y = y.transpose(1, 2)
        y_max_length = y.size(2)
        # norm speaker embeddings
        g = aux_input["d_vectors"] if aux_input is not None and "d_vectors" in aux_input else None
        if g is not None:
            if self.d_vector_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]

        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=g)
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)
        # find the alignment path
        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (o_mean ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
        attn = attn.squeeze(1).permute(0, 2, 1)
        outputs = {
            "model_outputs": z.transpose(1, 2),
            "logdet": logdet,
            "y_mean": y_mean.transpose(1, 2),
            "y_log_scale": y_log_scale.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dur_log.transpose(1, 2),
            "total_durations_log": o_attn_dur.transpose(1, 2),
        }
        return outputs

    @torch.no_grad()
    def inference_with_MAS(
        self, x, x_lengths, y=None, y_lengths=None, aux_input={"d_vectors": None}
    ):  # pylint: disable=dangerous-default-value
        """
        It's similar to the teacher forcing in Tacotron.
        It was proposed in: https://arxiv.org/abs/2104.05557

        Shapes:
            - x: :math:`[B, T]`
            - x_lenghts: :math:`B`
            - y: :math:`[B, T, C]`
            - y_lengths: :math:`B`
            - g: :math:`[B, C] or B`
        """
        y = y.transpose(1, 2)
        y_max_length = y.size(2)
        # norm speaker embeddings
        g = aux_input["d_vectors"] if aux_input is not None and "d_vectors" in aux_input else None
        if g is not None:
            if self.external_d_vector_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]

        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=g)
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)
        # find the alignment path between z and encoder output
        o_scale = torch.exp(-2 * o_log_scale)
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
        logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z ** 2))  # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
        logp4 = torch.sum(-0.5 * (o_mean ** 2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
        logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
        attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()

        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
        attn = attn.squeeze(1).permute(0, 2, 1)

        # get predited aligned distribution
        z = y_mean * y_mask

        # reverse the decoder and predict using the aligned distribution
        y, logdet = self.decoder(z, y_mask, g=g, reverse=True)
        outputs = {
            "model_outputs": z.transpose(1, 2),
            "logdet": logdet,
            "y_mean": y_mean.transpose(1, 2),
            "y_log_scale": y_log_scale.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dur_log.transpose(1, 2),
            "total_durations_log": o_attn_dur.transpose(1, 2),
        }
        return outputs

    @torch.no_grad()
    def decoder_inference(
        self, y, y_lengths=None, aux_input={"d_vectors": None}
    ):  # pylint: disable=dangerous-default-value
        """
        Shapes:
            - y: :math:`[B, T, C]`
            - y_lengths: :math:`B`
            - g: :math:`[B, C] or B`
        """
        y = y.transpose(1, 2)
        y_max_length = y.size(2)
        g = aux_input["d_vectors"] if aux_input is not None and "d_vectors" in aux_input else None
        # norm speaker embeddings
        if g is not None:
            if self.external_d_vector_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]

        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(y.dtype)

        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)

        # reverse decoder and predict
        y, logdet = self.decoder(z, y_mask, g=g, reverse=True)

        outputs = {}
        outputs["model_outputs"] = y.transpose(1, 2)
        outputs["logdet"] = logdet
        return outputs

    @torch.no_grad()
    def inference(self, x, aux_input={"x_lengths": None, "d_vectors": None}):  # pylint: disable=dangerous-default-value
        x_lengths = aux_input["x_lengths"]
        g = aux_input["d_vectors"] if aux_input is not None and "d_vectors" in aux_input else None

        if g is not None:
            if self.d_vector_dim:
                g = F.normalize(g).unsqueeze(-1)
            else:
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h]

        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=g)
        # compute output durations
        w = (torch.exp(o_dur_log) - 1) * x_mask * self.length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = None
        # compute masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # compute attention mask
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)

        z = (y_mean + torch.exp(y_log_scale) * torch.randn_like(y_mean) * self.inference_noise_scale) * y_mask
        # decoder pass
        y, logdet = self.decoder(z, y_mask, g=g, reverse=True)
        attn = attn.squeeze(1).permute(0, 2, 1)
        outputs = {
            "model_outputs": y.transpose(1, 2),
            "logdet": logdet,
            "y_mean": y_mean.transpose(1, 2),
            "y_log_scale": y_log_scale.transpose(1, 2),
            "alignments": attn,
            "durations_log": o_dur_log.transpose(1, 2),
            "total_durations_log": o_attn_dur.transpose(1, 2),
        }
        return outputs

    def train_step(self, batch: dict, criterion: nn.Module):
        """Perform a single training step by fetching the right set if samples from the batch.

        Args:
            batch (dict): [description]
            criterion (nn.Module): [description]
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        d_vectors = batch["d_vectors"]

        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input={"d_vectors": d_vectors})

        loss_dict = criterion(
            outputs["model_outputs"],
            outputs["y_mean"],
            outputs["y_log_scale"],
            outputs["logdet"],
            mel_lengths,
            outputs["durations_log"],
            outputs["total_durations_log"],
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

    def preprocess(self, y, y_lengths, y_max_length, attn=None):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.num_squeeze) * self.num_squeeze
            y = y[:, :, :y_max_length]
            if attn is not None:
                attn = attn[:, :, :, :y_max_length]
        y_lengths = (y_lengths // self.num_squeeze) * self.num_squeeze
        return y, y_lengths, y_max_length, attn

    def store_inverse(self):
        self.decoder.store_inverse()

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            self.store_inverse()
            assert not self.training

    def get_criterion(self):
        from TTS.tts.layers.losses import GlowTTSLoss  # pylint: disable=import-outside-toplevel

        return GlowTTSLoss()
