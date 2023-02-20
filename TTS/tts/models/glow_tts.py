import math
from typing import Dict, List, Tuple, Union

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from torch.nn import functional as F

from TTS.tts.configs.glow_tts_config import GlowTTSConfig
from TTS.tts.layers.glow_tts.decoder import Decoder
from TTS.tts.layers.glow_tts.encoder import Encoder
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import generate_path, maximum_path, sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.io import load_fsspec


class GlowTTS(BaseTTS):
    """GlowTTS model.

    Paper::
        https://arxiv.org/abs/2005.11129

    Paper abstract::
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
        Init only model layers.

        >>> from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        >>> from TTS.tts.models.glow_tts import GlowTTS
        >>> config = GlowTTSConfig(num_chars=2)
        >>> model = GlowTTS(config)

        Fully init a model ready for action. All the class attributes and class members
        (e.g Tokenizer, AudioProcessor, etc.). are initialized internally based on config values.

        >>> from TTS.tts.configs.glow_tts_config import GlowTTSConfig
        >>> from TTS.tts.models.glow_tts import GlowTTS
        >>> config = GlowTTSConfig()
        >>> model = GlowTTS.init_from_config(config, verbose=False)
    """

    def __init__(
        self,
        config: GlowTTSConfig,
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        # pass all config fields to `self`
        # for fewer code change
        self.config = config
        for key in config:
            setattr(self, key, config[key])

        self.decoder_output_dim = config.out_channels

        # init multi-speaker layers if necessary
        self.init_multispeaker(config)

        self.run_data_dep_init = config.data_dep_init_steps > 0
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

    def init_multispeaker(self, config: Coqpit):
        """Init speaker embedding layer if `use_speaker_embedding` is True and set the expected speaker embedding
        vector dimension to the encoder layer channel size. If model uses d-vectors, then it only sets
        speaker embedding vector dimension to the d-vector dimension from the config.

        Args:
            config (Coqpit): Model configuration.
        """
        self.embedded_speaker_dim = 0
        # set number of speakers - if num_speakers is set in config, use it, otherwise use speaker_manager
        if self.speaker_manager is not None:
            self.num_speakers = self.speaker_manager.num_speakers
        # set ultimate speaker embedding size
        if config.use_d_vector_file:
            self.embedded_speaker_dim = (
                config.d_vector_dim if "d_vector_dim" in config and config.d_vector_dim is not None else 512
            )
            if self.speaker_manager is not None:
                assert (
                    config.d_vector_dim == self.speaker_manager.embedding_dim
                ), " [!] d-vector dimension mismatch b/w config and speaker manager."
        # init speaker embedding layer
        if config.use_speaker_embedding and not config.use_d_vector_file:
            print(" > Init speaker_embedding layer.")
            self.embedded_speaker_dim = self.hidden_channels_enc
            self.emb_g = nn.Embedding(self.num_speakers, self.hidden_channels_enc)
            nn.init.uniform_(self.emb_g.weight, -0.1, 0.1)
        # set conditioning dimensions
        self.c_in_channels = self.embedded_speaker_dim

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

    def unlock_act_norm_layers(self):
        """Unlock activation normalization layers for data depended initalization."""
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(True)

    def lock_act_norm_layers(self):
        """Lock activation normalization layers."""
        for f in self.decoder.flows:
            if getattr(f, "set_ddi", False):
                f.set_ddi(False)

    def _set_speaker_input(self, aux_input: Dict):
        if aux_input is None:
            d_vectors = None
            speaker_ids = None
        else:
            d_vectors = aux_input.get("d_vectors", None)
            speaker_ids = aux_input.get("speaker_ids", None)

        if d_vectors is not None and speaker_ids is not None:
            raise ValueError("[!] Cannot use d-vectors and speaker-ids together.")

        if speaker_ids is not None and not hasattr(self, "emb_g"):
            raise ValueError("[!] Cannot use speaker-ids without enabling speaker embedding.")

        g = speaker_ids if speaker_ids is not None else d_vectors
        return g

    def _speaker_embedding(self, aux_input: Dict) -> Union[torch.tensor, None]:
        g = self._set_speaker_input(aux_input)
        # speaker embedding
        if g is not None:
            if hasattr(self, "emb_g"):
                # use speaker embedding layer
                if not g.size():  # if is a scalar
                    g = g.unsqueeze(0)  # unsqueeze
                g = F.normalize(self.emb_g(g)).unsqueeze(-1)  # [b, h, 1]
            else:
                # use d-vector
                g = F.normalize(g).unsqueeze(-1)  # [b, h, 1]
        return g

    def forward(
        self, x, x_lengths, y, y_lengths=None, aux_input={"d_vectors": None, "speaker_ids": None}
    ):  # pylint: disable=dangerous-default-value
        """
        Args:
            x (torch.Tensor):
                Input text sequence ids. :math:`[B, T_en]`

            x_lengths (torch.Tensor):
                Lengths of input text sequences. :math:`[B]`

            y (torch.Tensor):
                Target mel-spectrogram frames. :math:`[B, T_de, C_mel]`

            y_lengths (torch.Tensor):
                Lengths of target mel-spectrogram frames. :math:`[B]`

            aux_input (Dict):
                Auxiliary inputs. `d_vectors` is speaker embedding vectors for a multi-speaker model.
                :math:`[B, D_vec]`. `speaker_ids` is speaker ids for a multi-speaker model usind speaker-embedding
                layer. :math:`B`

        Returns:
            Dict:
                - z: :math: `[B, T_de, C]`
                - logdet: :math:`B`
                - y_mean: :math:`[B, T_de, C]`
                - y_log_scale: :math:`[B, T_de, C]`
                - alignments: :math:`[B, T_en, T_de]`
                - durations_log: :math:`[B, T_en, 1]`
                - total_durations_log: :math:`[B, T_en, 1]`
        """
        # [B, T, C] -> [B, C, T]
        y = y.transpose(1, 2)
        y_max_length = y.size(2)
        # norm speaker embeddings
        g = self._speaker_embedding(aux_input)
        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=g)
        # drop redisual frames wrt num_squeeze and set y_lengths.
        y, y_lengths, y_max_length, attn = self.preprocess(y, y_lengths, y_max_length, None)
        # create masks
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, y_max_length), 1).to(x_mask.dtype)
        # [B, 1, T_en, T_de]
        attn_mask = torch.unsqueeze(x_mask, -1) * torch.unsqueeze(y_mask, 2)
        # decoder pass
        z, logdet = self.decoder(y, y_mask, g=g, reverse=False)
        # find the alignment path
        with torch.no_grad():
            o_scale = torch.exp(-2 * o_log_scale)
            logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - o_log_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
            logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
            logp = logp1 + logp2 + logp3 + logp4  # [b, t, t']
            attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()
        y_mean, y_log_scale, o_attn_dur = self.compute_outputs(attn, o_mean, o_log_scale, x_mask)
        attn = attn.squeeze(1).permute(0, 2, 1)
        outputs = {
            "z": z.transpose(1, 2),
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
        self, x, x_lengths, y=None, y_lengths=None, aux_input={"d_vectors": None, "speaker_ids": None}
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
        g = self._speaker_embedding(aux_input)
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
        logp2 = torch.matmul(o_scale.transpose(1, 2), -0.5 * (z**2))  # [b, t, d] x [b, d, t'] = [b, t, t']
        logp3 = torch.matmul((o_mean * o_scale).transpose(1, 2), z)  # [b, t, d] x [b, d, t'] = [b, t, t']
        logp4 = torch.sum(-0.5 * (o_mean**2) * o_scale, [1]).unsqueeze(-1)  # [b, t, 1]
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
        self, y, y_lengths=None, aux_input={"d_vectors": None, "speaker_ids": None}
    ):  # pylint: disable=dangerous-default-value
        """
        Shapes:
            - y: :math:`[B, T, C]`
            - y_lengths: :math:`B`
            - g: :math:`[B, C] or B`
        """
        y = y.transpose(1, 2)
        y_max_length = y.size(2)
        g = self._speaker_embedding(aux_input)
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
    def inference(
        self, x, aux_input={"x_lengths": None, "d_vectors": None, "speaker_ids": None}
    ):  # pylint: disable=dangerous-default-value
        x_lengths = aux_input["x_lengths"]
        g = self._speaker_embedding(aux_input)
        # embedding pass
        o_mean, o_log_scale, o_dur_log, x_mask = self.encoder(x, x_lengths, g=g)
        # compute output durations
        w = (torch.exp(o_dur_log) - 1) * x_mask * self.length_scale
        w_ceil = torch.clamp_min(torch.ceil(w), 1)
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
        """A single training step. Forward pass and loss computation. Run data depended initialization for the
        first `config.data_dep_init_steps` steps.

        Args:
            batch (dict): [description]
            criterion (nn.Module): [description]
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        d_vectors = batch["d_vectors"]
        speaker_ids = batch["speaker_ids"]

        if self.run_data_dep_init and self.training:
            # compute data-dependent initialization of activation norm layers
            self.unlock_act_norm_layers()
            with torch.no_grad():
                _ = self.forward(
                    text_input,
                    text_lengths,
                    mel_input,
                    mel_lengths,
                    aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids},
                )
            outputs = None
            loss_dict = None
            self.lock_act_norm_layers()
        else:
            # normal training step
            outputs = self.forward(
                text_input,
                text_lengths,
                mel_input,
                mel_lengths,
                aux_input={"d_vectors": d_vectors, "speaker_ids": speaker_ids},
            )

            with autocast(enabled=False):  # avoid mixed_precision in criterion
                loss_dict = criterion(
                    outputs["z"].float(),
                    outputs["y_mean"].float(),
                    outputs["y_log_scale"].float(),
                    outputs["logdet"].float(),
                    mel_lengths,
                    outputs["durations_log"].float(),
                    outputs["total_durations_log"].float(),
                    text_lengths,
                )
        return outputs, loss_dict

    def _create_logs(self, batch, outputs, ap):
        alignments = outputs["alignments"]
        text_input = batch["text_input"][:1] if batch["text_input"] is not None else None
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        d_vectors = batch["d_vectors"][:1] if batch["d_vectors"] is not None else None
        speaker_ids = batch["speaker_ids"][:1] if batch["speaker_ids"] is not None else None

        # model runs reverse flow to predict spectrograms
        pred_outputs = self.inference(
            text_input,
            aux_input={"x_lengths": text_lengths[:1], "d_vectors": d_vectors, "speaker_ids": speaker_ids},
        )
        model_outputs = pred_outputs["model_outputs"]

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

    @torch.no_grad()
    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    @torch.no_grad()
    def test_run(self, assets: Dict) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        if len(test_sentences) == 0:
            print(" | [!] No test sentences provided.")
        else:
            for idx, sen in enumerate(test_sentences):
                outputs = synthesis(
                    self,
                    sen,
                    self.config,
                    "cuda" in str(next(self.parameters()).device),
                    speaker_id=aux_inputs["speaker_id"],
                    d_vector=aux_inputs["d_vector"],
                    style_wav=aux_inputs["style_wav"],
                    use_griffin_lim=True,
                    do_trim_silence=False,
                )

                test_audios["{}-audio".format(idx)] = outputs["wav"]
                test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                    outputs["outputs"]["model_outputs"], self.ap, output_fig=False
                )
                test_figures["{}-alignment".format(idx)] = plot_alignment(outputs["alignments"], output_fig=False)
        return test_figures, test_audios

    def preprocess(self, y, y_lengths, y_max_length, attn=None):
        if y_max_length is not None:
            y_max_length = (y_max_length // self.num_squeeze) * self.num_squeeze
            y = y[:, :, :y_max_length]
            if attn is not None:
                attn = attn[:, :, :, :y_max_length]
        y_lengths = torch.div(y_lengths, self.num_squeeze, rounding_mode="floor") * self.num_squeeze
        return y, y_lengths, y_max_length, attn

    def store_inverse(self):
        self.decoder.store_inverse()

    def load_checkpoint(
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if eval:
            self.eval()
            self.store_inverse()
            assert not self.training

    @staticmethod
    def get_criterion():
        from TTS.tts.layers.losses import GlowTTSLoss  # pylint: disable=import-outside-toplevel

        return GlowTTSLoss()

    def on_train_step_start(self, trainer):
        """Decide on every training step wheter enable/disable data depended initialization."""
        self.run_data_dep_init = trainer.total_steps_done < self.data_dep_init_steps

    @staticmethod
    def init_from_config(config: "GlowTTSConfig", samples: Union[List[List], List[Dict]] = None, verbose=True):
        """Initiate model from config

        Args:
            config (VitsConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
            verbose (bool): If True, print init messages. Defaults to True.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config, verbose)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config, samples)
        return GlowTTS(new_config, ap, tokenizer, speaker_manager)
