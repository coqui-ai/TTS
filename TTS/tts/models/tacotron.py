# coding: utf-8

from typing import Dict, List, Tuple

import torch
from coqpit import Coqpit
from torch import nn
from torch.cuda.amp.autocast_mode import autocast

from TTS.tts.layers.tacotron.capacitron_layers import CapacitronVAE
from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.tts.layers.tacotron.tacotron import Decoder, Encoder, PostCBHG
from TTS.tts.models.base_tacotron import BaseTacotron
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.speakers import SpeakerManager
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.trainer_utils import get_optimizer


class Tacotron(BaseTacotron):
    """Tacotron as in https://arxiv.org/abs/1703.10135
    It's an autoregressive encoder-attention-decoder-postnet architecture.
    Check `TacotronConfig` for the arguments.

    Args:
        config (TacotronConfig): Configuration for the Tacotron model.
        speaker_manager (SpeakerManager): Speaker manager to handle multi-speaker settings. Only use if the model is
            a multi-speaker model. Defaults to None.
    """

    def __init__(self, config: Coqpit, speaker_manager: SpeakerManager = None):
        super().__init__(config)

        self.speaker_manager = speaker_manager
        chars, self.config, _ = self.get_characters(config)
        config.num_chars = self.num_chars = len(chars)

        # pass all config fields to `self`
        # for fewer code change
        for key in config:
            setattr(self, key, config[key])

        # set speaker embedding channel size for determining `in_channels` for the connected layers.
        # `init_multispeaker` needs to be called once more in training to initialize the speaker embedding layer based
        # on the number of speakers infered from the dataset.
        if self.use_speaker_embedding or self.use_d_vector_file:
            self.init_multispeaker(config)
            self.decoder_in_features += self.embedded_speaker_dim  # add speaker embedding dim

        if self.use_gst:
            self.decoder_in_features += self.gst.gst_embedding_dim

        if self.use_capacitron_vae:
            self.decoder_in_features += self.capacitron_vae.capacitron_VAE_embedding_dim

        # embedding layer
        self.embedding = nn.Embedding(self.num_chars, 256, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.3)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)
        self.decoder = Decoder(
            self.decoder_in_features,
            self.decoder_output_dim,
            self.r,
            self.memory_size,
            self.attention_type,
            self.windowing,
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
        self.postnet = PostCBHG(self.decoder_output_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2, self.out_channels)

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

        # Capacitron layers
        if self.capacitron_vae and self.use_capacitron_vae:
            self.capacitron_vae_layer = CapacitronVAE(
                num_mel=self.decoder_output_dim,
                encoder_output_dim=self.encoder_in_features,
                capacitron_VAE_embedding_dim=self.capacitron_vae.capacitron_VAE_embedding_dim,
                speaker_embedding_dim=self.embedded_speaker_dim
                if self.use_speaker_embedding and self.capacitron_vae.capacitron_use_speaker_embedding
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
                self.memory_size,
                self.attention_type,
                self.windowing,
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

    def forward(  # pylint: disable=dangerous-default-value
        self, text, text_lengths, mel_specs=None, mel_lengths=None, aux_input={"speaker_ids": None, "d_vectors": None}
    ):
        """
        Shapes:
            text: [B, T_in]
            text_lengths: [B]
            mel_specs: [B, T_out, C]
            mel_lengths: [B]
            aux_input: 'speaker_ids': [B, 1] and  'd_vectors':[B, C]
        """
        aux_input = self._format_aux_input(aux_input)
        outputs = {"alignments_backward": None, "decoder_outputs_backward": None}
        inputs = self.embedding(text)
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x T_in x encoder_in_features
        encoder_outputs = self.encoder(inputs)
        # sequence masking
        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)
        # global style token
        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)
        # speaker embedding
        if self.use_speaker_embedding or self.use_d_vector_file:
            if not self.use_d_vector_file:
                # B x 1 x speaker_embed_dim
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])[:, None]
            else:
                # B x 1 x speaker_embed_dim
                embedded_speakers = torch.unsqueeze(aux_input["d_vectors"], 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)
        # Capacitron
        if self.capacitron_vae and self.use_capacitron_vae:
            # B x capacitron_VAE_embedding_dim
            encoder_outputs, *capacitron_vae_outputs = self.compute_capacitron_VAE_embedding(
                encoder_outputs,
                reference_mel_info=[mel_specs, mel_lengths],
                text_info=[inputs, text_lengths]
                if self.capacitron_vae.capacitron_use_text_summary_embeddings
                else None,
                speaker_embedding=embedded_speakers if self.capacitron_vae.capacitron_use_speaker_embedding else None,
            )
        else:
            capacitron_vae_outputs = None
        # decoder_outputs: B x decoder_in_features x T_out
        # alignments: B x T_in x encoder_in_features
        # stop_tokens: B x T_in
        decoder_outputs, alignments, stop_tokens = self.decoder(encoder_outputs, mel_specs, input_mask)
        # sequence masking
        if output_mask is not None:
            decoder_outputs = decoder_outputs * output_mask.unsqueeze(1).expand_as(decoder_outputs)
        # B x T_out x decoder_in_features
        postnet_outputs = self.postnet(decoder_outputs)
        # sequence masking
        if output_mask is not None:
            postnet_outputs = postnet_outputs * output_mask.unsqueeze(2).expand_as(postnet_outputs)
        # B x T_out x posnet_dim
        postnet_outputs = self.last_linear(postnet_outputs)
        # B x T_out x decoder_in_features
        decoder_outputs = decoder_outputs.transpose(1, 2).contiguous()
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
    def inference(self, text_input, aux_input=None):
        aux_input = self._format_aux_input(aux_input)
        inputs = self.embedding(text_input)
        encoder_outputs = self.encoder(inputs)
        if self.gst and self.use_gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, aux_input["style_mel"], aux_input["d_vectors"])
        if self.capacitron_vae and self.use_capacitron_vae:
            if aux_input["reference_text"] is not None:
                reference_text_embedding = self.embedding(aux_input["reference_text"])
                reference_text_length = torch.tensor([reference_text_embedding.size(1)], dtype=torch.int64).to(
                    encoder_outputs.device
                )  # pylint: disable=not-callable
            reference_mel_length = (
                torch.tensor([aux_input["reference_mel"].size(1)], dtype=torch.int64).to(encoder_outputs.device)
                if aux_input["reference_mel"] is not None
                else None
            )  # pylint: disable=not-callable
            # B x capacitron_VAE_embedding_dim
            encoder_outputs, *_ = self.compute_capacitron_VAE_embedding(
                encoder_outputs,
                reference_mel_info=[aux_input["reference_mel"], reference_mel_length]
                if aux_input["reference_mel"] is not None
                else None,
                text_info=[reference_text_embedding, reference_text_length]
                if aux_input["reference_text"] is not None
                else None,
                speaker_embedding=aux_input["d_vectors"]
                if self.capacitron_vae.capacitron_use_speaker_embedding
                else None,
            )
        if self.num_speakers > 1:
            if not self.use_d_vector_file:
                # B x 1 x speaker_embed_dim
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])
                # reshape embedded_speakers
                if embedded_speakers.ndim == 1:
                    embedded_speakers = embedded_speakers[None, None, :]
                elif embedded_speakers.ndim == 2:
                    embedded_speakers = embedded_speakers[None, :]
            else:
                # B x 1 x speaker_embed_dim
                embedded_speakers = torch.unsqueeze(aux_input["d_vectors"], 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)
        decoder_outputs, alignments, stop_tokens = self.decoder.inference(encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = self.last_linear(postnet_outputs)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        outputs = {
            "model_outputs": postnet_outputs,
            "decoder_outputs": decoder_outputs,
            "alignments": alignments,
            "stop_tokens": stop_tokens,
        }
        return outputs

    def train_step(self, batch: Dict, criterion: torch.nn.Module) -> Tuple[Dict, Dict]:
        """Perform a single training step by fetching the right set of samples from the batch.

        Args:
            batch ([Dict]): A dictionary of input tensors.
            criterion ([torch.nn.Module]): Callable criterion to compute model loss.
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        linear_input = batch["linear_input"]
        stop_targets = batch["stop_targets"]
        stop_target_lengths = batch["stop_target_lengths"]
        speaker_ids = batch["speaker_ids"]
        d_vectors = batch["d_vectors"]

        # forward pass model
        outputs = self.forward(
            text_input,
            text_lengths,
            mel_input,
            mel_lengths,
            aux_input={"speaker_ids": speaker_ids, "d_vectors": d_vectors},
        )

        # set the [alignment] lengths wrt reduction factor for guided attention
        if mel_lengths.max() % self.decoder.r != 0:
            alignment_lengths = (
                mel_lengths + (self.decoder.r - (mel_lengths.max() % self.decoder.r))
            ) // self.decoder.r
        else:
            alignment_lengths = mel_lengths // self.decoder.r

        aux_input = {"speaker_ids": speaker_ids, "d_vectors": d_vectors}
        outputs = self.forward(text_input, text_lengths, mel_input, mel_lengths, aux_input)

        # compute loss
        with autocast(enabled=False):  # use float32 for the criterion
            loss_dict = criterion(
                outputs["model_outputs"].float(),
                outputs["decoder_outputs"].float(),
                mel_input.float(),
                linear_input.float(),
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
            # Initiate and return optimizers for the Capacitron VAE.
            # It returns 2 optimizers in a list. First one is for the general model,
            # the second one is for the single Lagrange multiplier-like variable Beta
            # Returns:
            #     List: optimizers.

            primary_params, secondary_params = self.capacitron_split_parameters()

            optimizer1 = get_optimizer(
                self.config.optimizer, self.config.optimizer_params, self.config.lr, parameters=primary_params
            )
            optimizer2 = get_optimizer(
                self.config.capacitron_vae.capacitron_secondary_optimizer,
                self.config.capacitron_vae.capacitron_secondary_optimizer_params,
                self.config.capacitron_vae.capacitron_secondary_optimizer_lr,
                parameters=secondary_params,
            )
            return [optimizer1, optimizer2]
        return self.config.optimizer

    def capacitron_split_parameters(self):
        primary_params = []
        secondary_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                if name == "capacitron_vae_layer.beta":
                    secondary_params.append(param)
                else:
                    primary_params.append(param)
        return [iter(primary_params), iter(secondary_params)]

    def get_lr(self) -> List:
        """Set the initial learning rates for each optimizer.

        Returns:
            List: learning rates for each optimizer.
        """
        return [self.config.lr, self.config.capacitron_vae.capacitron_secondary_optimizer_lr]

    def _create_logs(self, batch, outputs, ap):
        postnet_outputs = outputs["model_outputs"]
        decoder_outputs = outputs["decoder_outputs"]
        alignments = outputs["alignments"]
        alignments_backward = outputs["alignments_backward"]
        mel_input = batch["mel_input"]
        linear_input = batch["linear_input"]

        pred_linear_spec = postnet_outputs[0].data.cpu().numpy()
        pred_mel_spec = decoder_outputs[0].data.cpu().numpy()
        gt_linear_spec = linear_input[0].data.cpu().numpy()
        gt_mel_spec = mel_input[0].data.cpu().numpy()
        align_img = alignments[0].data.cpu().numpy()

        figures = {
            "pred_linear_spec": plot_spectrogram(pred_linear_spec, ap, output_fig=False),
            "real_linear_spec": plot_spectrogram(gt_linear_spec, ap, output_fig=False),
            "pred_mel_spec": plot_spectrogram(pred_mel_spec, ap, output_fig=False),
            "real_mel_spec": plot_spectrogram(gt_mel_spec, ap, output_fig=False),
            "alignment": plot_alignment(align_img, output_fig=False),
        }

        if self.bidirectional_decoder or self.double_decoder_consistency:
            figures["alignment_backward"] = plot_alignment(alignments_backward[0].data.cpu().numpy(), output_fig=False)

        # Sample audio
        audio = ap.inv_spectrogram(pred_linear_spec.T)
        return figures, {"audio": audio}

    def train_log(
        self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int
    ) -> None:  # pylint: disable=no-self-use
        ap = assets["audio_processor"]
        figures, audios = self._create_logs(batch, outputs, ap)
        logger.train_figures(steps, figures)
        logger.train_audios(steps, audios, ap.sample_rate)

    def eval_step(self, batch: dict, criterion: nn.Module):
        return self.train_step(batch, criterion)

    def eval_log(self, batch: dict, outputs: dict, logger: "Logger", assets: dict, steps: int) -> None:
        ap = assets["audio_processor"]
        figures, audios = self._create_logs(batch, outputs, ap)
        logger.eval_figures(steps, figures)
        logger.eval_audios(steps, audios, ap.sample_rate)
