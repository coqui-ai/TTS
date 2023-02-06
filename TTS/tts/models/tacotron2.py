# coding: utf-8

from typing import Dict, List, Union

import torch
from torch import nn
from torch.cuda.amp.autocast_mode import autocast
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.layers.tacotron.capacitron_layers import CapacitronVAE
from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.tts.layers.tacotron.tacotron2 import Decoder, Encoder, Postnet
from TTS.tts.models.base_tacotron import BaseTacotron
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.capacitron_optimizer import CapacitronOptimizer


class Tacotron2(BaseTacotron):
    """Tacotron2 model implementation inherited from :class:`TTS.tts.models.base_tacotron.BaseTacotron`.

    Paper::
        https://arxiv.org/abs/1712.05884

    Paper abstract::
        This paper describes Tacotron 2, a neural network architecture for speech synthesis directly from text.
        The system is composed of a recurrent sequence-to-sequence feature prediction network that maps character
        embeddings to mel-scale spectrograms, followed by a modified WaveNet model acting as a vocoder to synthesize
        timedomain waveforms from those spectrograms. Our model achieves a mean opinion score (MOS) of 4.53 comparable
        to a MOS of 4.58 for professionally recorded speech. To validate our design choices, we present ablation
        studies of key components of our system and evaluate the impact of using mel spectrograms as the input to
        WaveNet instead of linguistic, duration, and F0 features. We further demonstrate that using a compact acoustic
        intermediate representation enables significant simplification of the WaveNet architecture.

    Check :class:`TTS.tts.configs.tacotron2_config.Tacotron2Config` for model arguments.

    Args:
        config (TacotronConfig):
            Configuration for the Tacotron2 model.
        speaker_manager (SpeakerManager):
            Speaker manager for multi-speaker training. Uuse only for multi-speaker training. Defaults to None.
    """

    def __init__(
        self,
        config: "Tacotron2Config",
        ap: "AudioProcessor" = None,
        tokenizer: "TTSTokenizer" = None,
        speaker_manager: SpeakerManager = None,
    ):
        super().__init__(config, ap, tokenizer, speaker_manager)

        self.decoder_output_dim = config.out_channels

        # pass all config fields to `self`
        # for fewer code change
        for key in config:
            setattr(self, key, config[key])

        # init multi-speaker layers
        if self.use_speaker_embedding or self.use_d_vector_file:
            self.init_multispeaker(config)
            self.decoder_in_features += self.embedded_speaker_dim  # add speaker embedding dim

        if self.use_gst:
            self.decoder_in_features += self.gst.gst_embedding_dim

        if self.use_capacitron_vae:
            self.decoder_in_features += self.capacitron_vae.capacitron_VAE_embedding_dim

        # embedding layer
        self.embedding = nn.Embedding(self.num_chars, 512, padding_idx=0)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)

        self.decoder = Decoder(
            self.decoder_in_features,
            self.decoder_output_dim,
            self.r,
            self.attention_type,
            self.attention_win,
            self.attention_norm,
            self.prenet_type,
            self.prenet_dropout,
            self.use_forward_attn,
            self.transition_agent,
            self.forward_attn_mask,
            self.location_attn,
            self.attention_heads,
            self.separate_stopnet,
            self.max_decoder_steps,
        )
        self.postnet = Postnet(self.out_channels)

        # setup prenet dropout
        self.decoder.prenet.dropout_at_inference = self.prenet_dropout_at_inference

        # global style token layers
        if self.gst and self.use_gst:
            self.gst_layer = GST(
                num_mel=self.decoder_output_dim,
                num_heads=self.gst.gst_num_heads,
                num_style_tokens=self.gst.gst_num_style_tokens,
                gst_embedding_dim=self.gst.gst_embedding_dim,
            )

        # Capacitron VAE Layers
        if self.capacitron_vae and self.use_capacitron_vae:
            self.capacitron_vae_layer = CapacitronVAE(
                num_mel=self.decoder_output_dim,
                encoder_output_dim=self.encoder_in_features,
                capacitron_VAE_embedding_dim=self.capacitron_vae.capacitron_VAE_embedding_dim,
                speaker_embedding_dim=self.embedded_speaker_dim
                if self.capacitron_vae.capacitron_use_speaker_embedding
                else None,
                text_summary_embedding_dim=self.capacitron_vae.capacitron_text_summary_embedding_dim
                if self.capacitron_vae.capacitron_use_text_summary_embeddings
                else None,
            )

        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                self.decoder_in_features,
                self.decoder_output_dim,
                self.ddc_r,
                self.attention_type,
                self.attention_win,
                self.attention_norm,
                self.prenet_type,
                self.prenet_dropout,
                self.use_forward_attn,
                self.transition_agent,
                self.forward_attn_mask,
                self.location_attn,
                self.attention_heads,
                self.separate_stopnet,
                self.max_decoder_steps,
            )

    @staticmethod
    def shape_outputs(mel_outputs, mel_outputs_postnet, alignments):
        """Final reshape of the model output tensors."""
        mel_outputs = mel_outputs.transpose(1, 2)
        mel_outputs_postnet = mel_outputs_postnet.transpose(1, 2)
        return mel_outputs, mel_outputs_postnet, alignments

    def forward(  # pylint: disable=dangerous-default-value
        self, text, text_lengths, mel_specs=None, mel_lengths=None, aux_input={"speaker_ids": None, "d_vectors": None}
    ):
        """Forward pass for training with Teacher Forcing.

        Shapes:
            text: :math:`[B, T_in]`
            text_lengths: :math:`[B]`
            mel_specs: :math:`[B, T_out, C]`
            mel_lengths: :math:`[B]`
            aux_input: 'speaker_ids': :math:`[B, 1]` and  'd_vectors': :math:`[B, C]`
        """
        aux_input = self._format_aux_input(aux_input)
        outputs = {"alignments_backward": None, "decoder_outputs_backward": None}
        # compute mask for padding
        # B x T_in_max (boolean)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x D_embed x T_in_max
        embedded_inputs = self.embedding(text).transpose(1, 2)
        # B x T_in_max x D_en
        encoder_outputs = self.encoder(embedded_inputs, text_lengths)
        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)

        if self.use_speaker_embedding or self.use_d_vector_file:
            if not self.use_d_vector_file:
                # B x 1 x speaker_embed_dim
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])[:, None]
            else:
                # B x 1 x speaker_embed_dim
                embedded_speakers = torch.unsqueeze(aux_input["d_vectors"], 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)

        # capacitron
        if self.capacitron_vae and self.use_capacitron_vae:
            # B x capacitron_VAE_embedding_dim
            encoder_outputs, *capacitron_vae_outputs = self.compute_capacitron_VAE_embedding(
                encoder_outputs,
                reference_mel_info=[mel_specs, mel_lengths],
                text_info=[embedded_inputs.transpose(1, 2), text_lengths]
                if self.capacitron_vae.capacitron_use_text_summary_embeddings
                else None,
                speaker_embedding=embedded_speakers if self.capacitron_vae.capacitron_use_speaker_embedding else None,
            )
        else:
            capacitron_vae_outputs = None

        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)

        # B x mel_dim x T_out -- B x T_out//r x T_in -- B x T_out//r
        decoder_outputs, alignments, stop_tokens = self.decoder(encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if mel_lengths is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs)
        # B x mel_dim x T_out
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        # sequence masking
        if output_mask is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(1).expand_as(postnet_outputs)
        # B x T_out x mel_dim -- B x T_out x mel_dim -- B x T_out//r x T_in
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(decoder_outputs, postnet_outputs, alignments)
        if self.bidirectional_decoder:
            decoder_outputs_backward, alignments_backward = self._backward_pass(mel_specs, encoder_outputs, input_mask)
            outputs["alignments_backward"] = alignments_backward
            outputs["decoder_outputs_backward"] = decoder_outputs_backward
        if self.double_decoder_consistency:
            decoder_outputs_backward, alignments_backward = self._coarse_decoder_pass(
                mel_specs, encoder_outputs, alignments, input_mask
            )
            outputs["alignments_backward"] = alignments_backward
            outputs["decoder_outputs_backward"] = decoder_outputs_backward
        outputs.update(
            {
                "model_outputs": postnet_outputs,
                "decoder_outputs": decoder_outputs,
                "alignments": alignments,
                "stop_tokens": stop_tokens,
                "capacitron_vae_outputs": capacitron_vae_outputs,
            }
        )
        return outputs

    @torch.no_grad()
    def inference(self, text, aux_input=None):
        """Forward pass for inference with no Teacher-Forcing.

        Shapes:
           text: :math:`[B, T_in]`
           text_lengths: :math:`[B]`
        """
        aux_input = self._format_aux_input(aux_input)
        embedded_inputs = self.embedding(text).transpose(1, 2)
        encoder_outputs = self.encoder.inference(embedded_inputs)

        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, aux_input["style_mel"], aux_input["d_vectors"])

        if self.capacitron_vae and self.use_capacitron_vae:
            if aux_input["style_text"] is not None:
                style_text_embedding = self.embedding(aux_input["style_text"])
                style_text_length = torch.tensor([style_text_embedding.size(1)], dtype=torch.int64).to(
                    encoder_outputs.device
                )  # pylint: disable=not-callable
            reference_mel_length = (
                torch.tensor([aux_input["style_mel"].size(1)], dtype=torch.int64).to(encoder_outputs.device)
                if aux_input["style_mel"] is not None
                else None
            )  # pylint: disable=not-callable
            # B x capacitron_VAE_embedding_dim
            encoder_outputs, *_ = self.compute_capacitron_VAE_embedding(
                encoder_outputs,
                reference_mel_info=[aux_input["style_mel"], reference_mel_length]
                if aux_input["style_mel"] is not None
                else None,
                text_info=[style_text_embedding, style_text_length] if aux_input["style_text"] is not None else None,
                speaker_embedding=aux_input["d_vectors"]
                if self.capacitron_vae.capacitron_use_speaker_embedding
                else None,
            )

        if self.num_speakers > 1:
            if not self.use_d_vector_file:
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])[None]
                # reshape embedded_speakers
                if embedded_speakers.ndim == 1:
                    embedded_speakers = embedded_speakers[None, None, :]
                elif embedded_speakers.ndim == 2:
                    embedded_speakers = embedded_speakers[None, :]
            else:
                embedded_speakers = aux_input["d_vectors"]

            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)

        decoder_outputs, alignments, stop_tokens = self.decoder.inference(encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = decoder_outputs + postnet_outputs
        decoder_outputs, postnet_outputs, alignments = self.shape_outputs(decoder_outputs, postnet_outputs, alignments)
        outputs = {
            "model_outputs": postnet_outputs,
            "decoder_outputs": decoder_outputs,
            "alignments": alignments,
            "stop_tokens": stop_tokens,
        }
        return outputs

    def before_backward_pass(self, loss_dict, optimizer) -> None:
        # Extracting custom training specific operations for capacitron
        # from the trainer
        if self.use_capacitron_vae:
            loss_dict["capacitron_vae_beta_loss"].backward()
            optimizer.first_step()

    def train_step(self, batch: Dict, criterion: torch.nn.Module):
        """A single training step. Forward pass and loss computation.

        Args:
            batch ([Dict]): A dictionary of input tensors.
            criterion ([type]): Callable criterion to compute model loss.
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        stop_targets = batch["stop_targets"]
        stop_target_lengths = batch["stop_target_lengths"]
        speaker_ids = batch["speaker_ids"]
        d_vectors = batch["d_vectors"]

        aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input)

        # set the [alignment] lengths wrt reduction factor for guided attention
        if mel_lengths.max() % self.decoder.r != 0:
            alignment_lengths = (
                mel_lengths + (self.decoder.r - (mel_lengths.max() % self.decoder.r))
            ) // self.decoder.r
        else:
            alignment_lengths = mel_lengths // self.decoder.r

        # compute loss
        with autocast(enabled=False):  # use float32 for the criterion
            loss_dict = criterion(
                outputs["model_outputs"].float(),
                outputs["decoder_outputs"].float(),
                mel_input.float(),
                None,
                outputs["stop_tokens"].float(),
                stop_targets.float(),
                stop_target_lengths,
                outputs["capacitron_vae_outputs"] if self.capacitron_vae else None,
                mel_lengths,
                None if outputs["decoder_outputs_backward"] is None else outputs["decoder_outputs_backward"].float(),
                outputs["alignments"].float(),
                alignment_lengths,
                None if outputs["alignments_backward"] is None else outputs["alignments_backward"].float(),
                text_lengths,
            )

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(outputs["alignments"])
        loss_dict["align_error"] = align_error
        return outputs, loss_dict

    def get_optimizer(self) -> List:
        if self.use_capacitron_vae:
            return CapacitronOptimizer(self.config, self.named_parameters())
        return get_optimizer(self.config.optimizer, self.config.optimizer_params, self.config.lr, self)

    def get_scheduler(self, optimizer: object):
        opt = optimizer.primary_optimizer if self.use_capacitron_vae else optimizer
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, opt)

    def before_gradient_clipping(self):
        if self.use_capacitron_vae:
            # Capacitron model specific gradient clipping
            model_params_to_clip = []
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name != "capacitron_vae_layer.beta":
                        model_params_to_clip.append(param)
            torch.nn.utils.clip_grad_norm_(model_params_to_clip, self.capacitron_vae.capacitron_grad_clip)

    def _create_logs(self, batch, outputs, ap):
        """Create dashboard log information."""
        postnet_outputs = outputs["model_outputs"]
        alignments = outputs["alignments"]
        alignments_backward = outputs["alignments_backward"]
        mel_input = batch["mel_input"]

        pred_spec = postnet_outputs[0].data.cpu().numpy()
        gt_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "prediction": plot_spectrogram(pred_spec, ap, output_fig=False),
            "ground_truth": plot_spectrogram(gt_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        if self.bidirectional_decoder or self.double_decoder_consistency:
            figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy(), output_fig=False)

        # Sample audio
        audio = ap.inv_melspectrogram(pred_spec.T)
        return figures, {"audio": audio}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        """Log training progress."""
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, self.ap.sample_rate)

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        figures, audios = self._create_logs(batch, outputs, self.ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, self.ap.sample_rate)

    @staticmethod
    def init_from_config(config: "Tacotron2Config", samples: Union[List[List], List[Dict]] = None):
        """Initiate model from config

        Args:
            config (Tacotron2Config): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        from TTS.utils.audio import AudioProcessor

        ap = AudioProcessor.init_from_config(config)
        tokenizer, new_config = TTSTokenizer.init_from_config(config)
        speaker_manager = SpeakerManager.init_from_config(new_config, samples)
        return Tacotron2(new_config, ap, tokenizer, speaker_manager)
