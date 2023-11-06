import copy
from abc import abstractmethod
from typing import Dict, Tuple

import torch
from coqpit import Coqpit
from torch import nn

from TTS.tts.layers.losses import TacotronLoss
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.helpers import sequence_mask
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.synthesis import synthesis
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.generic_utils import format_aux_input
from TTS.utils.io import load_fsspec
from TTS.utils.training import gradual_training_scheduler


class BaseTacotron(BaseTTS):
    """Base class shared by Tacotron and Tacotron2"""

    def __init__(
        self,
        config: "TacotronConfig",
        ap: "AudioProcessor",
        tokenizer: "TTSTokenizer",
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        # pass all config fields as class attributes
        for key in config:
            setattr(self, key, config[key])

        # layers
        self.embedding = None
        self.encoder = None
        self.decoder = None
        self.postnet = None

        # init tensors
        self.embedded_speakers = None
        self.embedded_speakers_projected = None

        # global style token
        if self.gst and self.use_gst:
            self.decoder_in_features += self.gst.gst_embedding_dim  # add gst embedding dim
            self.gst_layer = None

        # Capacitron
        if self.capacitron_vae and self.use_capacitron_vae:
            self.decoder_in_features += self.capacitron_vae.capacitron_VAE_embedding_dim  # add capacitron embedding dim
            self.capacitron_vae_layer = None

        # additional layers
        self.decoder_backward = None
        self.coarse_decoder = None

    @staticmethod
    def _format_aux_input(aux_input: Dict) -> Dict:
        """Set missing fields to their default values"""
        if aux_input:
            return format_aux_input({"d_vectors": None, "speaker_ids": None}, aux_input)
        return None

    #############################
    # INIT FUNCTIONS
    #############################

    def _init_backward_decoder(self):
        """Init the backward decoder for Forward-Backward decoding."""
        self.decoder_backward = copy.deepcopy(self.decoder)

    def _init_coarse_decoder(self):
        """Init the coarse decoder for Double-Decoder Consistency."""
        self.coarse_decoder = copy.deepcopy(self.decoder)
        self.coarse_decoder.r_init = self.ddc_r
        self.coarse_decoder.set_r(self.ddc_r)

    #############################
    # CORE FUNCTIONS
    #############################

    @abstractmethod
    def forward(self):
        pass

    @abstractmethod
    def inference(self):
        pass

    def load_checkpoint(
        self, config, checkpoint_path, eval=False, cache=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        """Load model checkpoint and set up internals.

        Args:
            config (Coqpi): model configuration.
            checkpoint_path (str): path to checkpoint file.
            eval (bool, optional): whether to load model for evaluation.
            cache (bool, optional): If True, cache the file locally for subsequent calls. It is cached under `get_user_data_dir()/tts_cache`. Defaults to False.
        """
        state = load_fsspec(checkpoint_path, map_location=torch.device("cpu"), cache=cache)
        self.load_state_dict(state["model"])
        # TODO: set r in run-time by taking it from the new config
        if "r" in state:
            # set r from the state (for compatibility with older checkpoints)
            self.decoder.set_r(state["r"])
        elif "config" in state:
            # set r from config used at training time (for inference)
            self.decoder.set_r(state["config"]["r"])
        else:
            # set r from the new config (for new-models)
            self.decoder.set_r(config.r)
        if eval:
            self.eval()
            print(f" > Model's reduction rate `r` is set to: {self.decoder.r}")
            assert not self.training

    def get_criterion(self) -> nn.Module:
        """Get the model criterion used in training."""
        return TacotronLoss(self.config)

    @staticmethod
    def init_from_config(config: Coqpit):
        """Initialize model from config."""
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(config)
        return BaseTacotron(config, ap, tokenizer, speaker_manager)

    ##########################
    # TEST AND LOG FUNCTIONS #
    ##########################

    def test_run(self, assets: Dict) -> Tuple[Dict, Dict]:
        """Generic test run for `tts` models used by `Trainer`.

        You can override this for a different behaviour.

        Args:
            assets (dict): A dict of training assets. For `tts` models, it must include `{'audio_processor': ap}`.

        Returns:
            Tuple[Dict, Dict]: Test figures and audios to be projected to Tensorboard.
        """
        print(" | > Synthesizing test sentences.")
        test_audios = {}
        test_figures = {}
        test_sentences = self.config.test_sentences
        aux_inputs = self._get_test_aux_input()
        for idx, sen in enumerate(test_sentences):
            outputs_dict = synthesis(
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
            test_audios["{}-audio".format(idx)] = outputs_dict["wav"]
            test_figures["{}-prediction".format(idx)] = plot_spectrogram(
                outputs_dict["outputs"]["model_outputs"], self.ap, output_fig=False
            )
            test_figures["{}-alignment".format(idx)] = plot_alignment(
                outputs_dict["outputs"]["alignments"], output_fig=False
            )
        return {"figures": test_figures, "audios": test_audios}

    def test_log(
        self, outputs: dict, logger: "Logger", assets: dict, steps: int  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.ap.sample_rate)
        logger.test_figures(steps, outputs["figures"])

    #############################
    # COMMON COMPUTE FUNCTIONS
    #############################

    def compute_masks(self, text_lengths, mel_lengths):
        """Compute masks  against sequence paddings."""
        # B x T_in_max (boolean)
        input_mask = sequence_mask(text_lengths)
        output_mask = None
        if mel_lengths is not None:
            max_len = mel_lengths.max()
            r = self.decoder.r
            max_len = max_len + (r - (max_len % r)) if max_len % r > 0 else max_len
            output_mask = sequence_mask(mel_lengths, max_len=max_len)
        return input_mask, output_mask

    def _backward_pass(self, mel_specs, encoder_outputs, mask):
        """Run backwards decoder"""
        decoder_outputs_b, alignments_b, _ = self.decoder_backward(
            encoder_outputs, torch.flip(mel_specs, dims=(1,)), mask
        )
        decoder_outputs_b = decoder_outputs_b.transpose(1, 2).contiguous()
        return decoder_outputs_b, alignments_b

    def _coarse_decoder_pass(self, mel_specs, encoder_outputs, alignments, input_mask):
        """Double Decoder Consistency"""
        T = mel_specs.shape[1]
        if T % self.coarse_decoder.r > 0:
            padding_size = self.coarse_decoder.r - (T % self.coarse_decoder.r)
            mel_specs = torch.nn.functional.pad(mel_specs, (0, 0, 0, padding_size, 0, 0))
        decoder_outputs_backward, alignments_backward, _ = self.coarse_decoder(
            encoder_outputs.detach(), mel_specs, input_mask
        )
        # scale_factor = self.decoder.r_init / self.decoder.r
        alignments_backward = torch.nn.functional.interpolate(
            alignments_backward.transpose(1, 2),
            size=alignments.shape[1],
            mode="nearest",
        ).transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward.transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward[:, :T, :]
        return decoder_outputs_backward, alignments_backward

    #############################
    # EMBEDDING FUNCTIONS
    #############################

    def compute_gst(self, inputs, style_input, speaker_embedding=None):
        """Compute global style token"""
        if isinstance(style_input, dict):
            # multiply each style token with a weight
            query = torch.zeros(1, 1, self.gst.gst_embedding_dim // 2).type_as(inputs)
            if speaker_embedding is not None:
                query = torch.cat([query, speaker_embedding.reshape(1, 1, -1)], dim=-1)

            _GST = torch.tanh(self.gst_layer.style_token_layer.style_tokens)
            gst_outputs = torch.zeros(1, 1, self.gst.gst_embedding_dim).type_as(inputs)
            for k_token, v_amplifier in style_input.items():
                key = _GST[int(k_token)].unsqueeze(0).expand(1, -1, -1)
                gst_outputs_att = self.gst_layer.style_token_layer.attention(query, key)
                gst_outputs = gst_outputs + gst_outputs_att * v_amplifier
        elif style_input is None:
            # ignore style token and return zero tensor
            gst_outputs = torch.zeros(1, 1, self.gst.gst_embedding_dim).type_as(inputs)
        else:
            # compute style tokens
            gst_outputs = self.gst_layer(style_input, speaker_embedding)  # pylint: disable=not-callable
        inputs = self._concat_speaker_embedding(inputs, gst_outputs)
        return inputs

    def compute_capacitron_VAE_embedding(self, inputs, reference_mel_info, text_info=None, speaker_embedding=None):
        """Capacitron Variational Autoencoder"""
        (
            VAE_outputs,
            posterior_distribution,
            prior_distribution,
            capacitron_beta,
        ) = self.capacitron_vae_layer(
            reference_mel_info,
            text_info,
            speaker_embedding,  # pylint: disable=not-callable
        )

        VAE_outputs = VAE_outputs.to(inputs.device)
        encoder_output = self._concat_speaker_embedding(
            inputs, VAE_outputs
        )  # concatenate to the output of the basic tacotron encoder
        return (
            encoder_output,
            posterior_distribution,
            prior_distribution,
            capacitron_beta,
        )

    @staticmethod
    def _add_speaker_embedding(outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = outputs + embedded_speakers_
        return outputs

    @staticmethod
    def _concat_speaker_embedding(outputs, embedded_speakers):
        embedded_speakers_ = embedded_speakers.expand(outputs.size(0), outputs.size(1), -1)
        outputs = torch.cat([outputs, embedded_speakers_], dim=-1)
        return outputs

    #############################
    # CALLBACKS
    #############################

    def on_epoch_start(self, trainer):
        """Callback for setting values wrt gradual training schedule.

        Args:
            trainer (TrainerTTS): TTS trainer object that is used to train this model.
        """
        if self.gradual_training:
            r, trainer.config.batch_size = gradual_training_scheduler(trainer.total_steps_done, trainer.config)
            trainer.config.r = r
            self.decoder.set_r(r)
            if trainer.config.bidirectional_decoder:
                trainer.model.decoder_backward.set_r(r)
            print(f"\n > Number of output frames: {self.decoder.r}")
