import copy
from abc import abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import torch
from coqpit import MISSING, Coqpit
from torch import nn

from TTS.tts.layers.losses import TacotronLoss
from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.utils.data import sequence_mask
from TTS.tts.utils.speakers import SpeakerManager, get_speaker_manager
from TTS.tts.utils.text import make_symbols
from TTS.utils.generic_utils import format_aux_input
from TTS.utils.training import gradual_training_scheduler


@dataclass
class BaseTacotronArgs(Coqpit):
    """TODO: update Tacotron configs using it"""

    num_chars: int = MISSING
    num_speakers: int = MISSING
    r: int = MISSING
    out_channels: int = 80
    decoder_output_dim: int = 80
    attn_type: str = "original"
    attn_win: bool = False
    attn_norm: str = "softmax"
    prenet_type: str = "original"
    prenet_dropout: bool = True
    prenet_dropout_at_inference: bool = False
    forward_attn: bool = False
    trans_agent: bool = False
    forward_attn_mask: bool = False
    location_attn: bool = True
    attn_K: int = 5
    separate_stopnet: bool = True
    bidirectional_decoder: bool = False
    double_decoder_consistency: bool = False
    ddc_r: int = None
    encoder_in_features: int = 512
    decoder_in_features: int = 512
    d_vector_dim: int = None
    use_gst: bool = False
    gst: bool = None
    gradual_training: bool = None


class BaseTacotron(BaseTTS):
    def __init__(self, config: Coqpit, data: List = None):
        """Abstract Tacotron class"""
        super().__init__()

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

        # additional layers
        self.decoder_backward = None
        self.coarse_decoder = None

        # init multi-speaker layers
        self.init_multispeaker(config, data)

    @staticmethod
    def _format_aux_input(aux_input: Dict) -> Dict:
        return format_aux_input({"d_vectors": None, "speaker_ids": None}, aux_input)

    #############################
    # INIT FUNCTIONS
    #############################

    def _init_states(self):
        self.embedded_speakers = None
        self.embedded_speakers_projected = None

    def _init_backward_decoder(self):
        self.decoder_backward = copy.deepcopy(self.decoder)

    def _init_coarse_decoder(self):
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
        self, config, checkpoint_path, eval=False
    ):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        self.load_state_dict(state["model"])
        if "r" in state:
            self.decoder.set_r(state["r"])
        else:
            self.decoder.set_r(state["config"]["r"])
        if eval:
            self.eval()
            assert not self.training

    def get_criterion(self) -> nn.Module:
        return TacotronLoss(self.config)

    @staticmethod
    def get_characters(config: Coqpit) -> str:
        # TODO: implement CharacterProcessor
        if config.characters is not None:
            symbols, phonemes = make_symbols(**config.characters)
        else:
            from TTS.tts.utils.text.symbols import (  # pylint: disable=import-outside-toplevel
                parse_symbols,
                phonemes,
                symbols,
            )

            config.characters = parse_symbols()
        model_characters = phonemes if config.use_phonemes else symbols
        return model_characters, config

    @staticmethod
    def get_speaker_manager(config: Coqpit, restore_path: str, data: List, out_path: str = None) -> SpeakerManager:
        return get_speaker_manager(config, restore_path, data, out_path)

    def get_aux_input(self, **kwargs) -> Dict:
        """Compute Tacotron's auxiliary inputs based on model config.
        - speaker d_vector
        - style wav for GST
        - speaker ID for speaker embedding
        """
        # setup speaker_id
        if self.config.use_speaker_embedding:
            speaker_id = kwargs.get("speaker_id", 0)
        else:
            speaker_id = None
        # setup d_vector
        d_vector = (
            self.speaker_manager.get_d_vectors_by_speaker(self.speaker_manager.speaker_names[0])
            if self.config.use_d_vector_file and self.config.use_speaker_embedding
            else None
        )
        # setup style_mel
        if "style_wav" in kwargs:
            style_wav = kwargs["style_wav"]
        elif self.config.has("gst_style_input"):
            style_wav = self.config.gst_style_input
        else:
            style_wav = None
        if style_wav is None and "use_gst" in self.config and self.config.use_gst:
            # inicialize GST with zero dict.
            style_wav = {}
            print("WARNING: You don't provided a gst style wav, for this reason we use a zero tensor!")
            for i in range(self.config.gst["gst_num_style_tokens"]):
                style_wav[str(i)] = 0
        aux_inputs = {"speaker_id": speaker_id, "style_wav": style_wav, "d_vector": d_vector}
        return aux_inputs

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
            alignments_backward.transpose(1, 2), size=alignments.shape[1], mode="nearest"
        ).transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward.transpose(1, 2)
        decoder_outputs_backward = decoder_outputs_backward[:, :T, :]
        return decoder_outputs_backward, alignments_backward

    #############################
    # EMBEDDING FUNCTIONS
    #############################

    def compute_speaker_embedding(self, speaker_ids):
        """Compute speaker embedding vectors"""
        if hasattr(self, "speaker_embedding") and speaker_ids is None:
            raise RuntimeError(" [!] Model has speaker embedding layer but speaker_id is not provided")
        if hasattr(self, "speaker_embedding") and speaker_ids is not None:
            self.embedded_speakers = self.speaker_embedding(speaker_ids).unsqueeze(1)
        if hasattr(self, "speaker_project_mel") and speaker_ids is not None:
            self.embedded_speakers_projected = self.speaker_project_mel(self.embedded_speakers).squeeze(1)

    def compute_gst(self, inputs, style_input, speaker_embedding=None):
        """Compute global style token"""
        if isinstance(style_input, dict):
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
            gst_outputs = torch.zeros(1, 1, self.gst.gst_embedding_dim).type_as(inputs)
        else:
            gst_outputs = self.gst_layer(style_input, speaker_embedding)  # pylint: disable=not-callable
        inputs = self._concat_speaker_embedding(inputs, gst_outputs)
        return inputs

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
