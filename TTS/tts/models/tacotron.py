# coding: utf-8

from typing import Dict, Tuple

import torch
from coqpit import Coqpit
from torch import nn

from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.tts.layers.tacotron.tacotron import Decoder, Encoder, PostCBHG
from TTS.tts.models.base_tacotron import BaseTacotron
from TTS.tts.utils.measures import alignment_diagonal_score
from TTS.tts.utils.visual import plot_alignment, plot_spectrogram
from TTS.utils.audio import AudioProcessor


class Tacotron(BaseTacotron):
    """Tacotron as in https://arxiv.org/abs/1703.10135
    It's an autoregressive encoder-attention-decoder-postnet architecture.
    Check `TacotronConfig` for the arguments.
    """

    def __init__(self, config: Coqpit, data):
        super().__init__(config, data)

        self.num_chars, self.config = self.get_characters(config)

        # pass all config fields to `self`
        # for fewer code change
        for key in config:
            setattr(self, key, config[key])

        # speaker and gst embeddings is concat in decoder input
        if self.num_speakers > 1:
            self.decoder_in_features += self.embedded_speaker_dim  # add speaker embedding dim

        if self.use_gst:
            self.decoder_in_features += self.gst.gst_embedding_dim

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
                d_vector_dim=self.d_vector_dim
                if self.config.gst.gst_use_speaker_embedding and self.use_speaker_embedding
                else None,
                num_heads=self.gst.gst_num_heads,
                num_style_tokens=self.gst.gst_num_style_tokens,
                gst_embedding_dim=self.gst.gst_embedding_dim,
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

    def forward(self, text, text_lengths, mel_specs=None, mel_lengths=None, aux_input=None):
        """
        Shapes:
            text: [B, T_in]
            text_lengths: [B]
            mel_specs: [B, T_out, C]
            mel_lengths: [B]
            aux_input: 'speaker_ids': [B, 1] and  'd_vectors':[B, C]
        """
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
            encoder_outputs = self.compute_gst(
                encoder_outputs, mel_specs, aux_input["d_vectors"] if "d_vectors" in aux_input else None
            )
        # speaker embedding
        if self.num_speakers > 1:
            if not self.use_d_vector_file:
                # B x 1 x speaker_embed_dim
                embedded_speakers = self.speaker_embedding(aux_input["speaker_ids"])[:, None]
            else:
                # B x 1 x speaker_embed_dim
                embedded_speakers = torch.unsqueeze(aux_input["d_vectors"], 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, embedded_speakers)
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

    def train_step(self, batch, criterion):
        """Perform a single training step by fetching the right set if samples from the batch.

        Args:
            batch ([type]): [description]
            criterion ([type]): [description]
        """
        text_input = batch["text_input"]
        text_lengths = batch["text_lengths"]
        mel_input = batch["mel_input"]
        mel_lengths = batch["mel_lengths"]
        linear_input = batch["linear_input"]
        stop_targets = batch["stop_targets"]
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
        loss_dict = criterion(
            outputs["model_outputs"],
            outputs["decoder_outputs"],
            mel_input,
            linear_input,
            outputs["stop_tokens"],
            stop_targets,
            mel_lengths,
            outputs["decoder_outputs_backward"],
            outputs["alignments"],
            alignment_lengths,
            outputs["alignments_backward"],
            text_lengths,
        )

        # compute alignment error (the lower the better )
        align_error = 1 - alignment_diagonal_score(outputs["alignments"])
        loss_dict["align_error"] = align_error
        return outputs, loss_dict

    def train_log(self, ap: AudioProcessor, batch: dict, outputs: dict) -> Tuple[Dict, Dict]:
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
        train_audio = ap.inv_spectrogram(pred_spec.T)
        return figures, {"audio": train_audio}

    def eval_step(self, batch, criterion):
        return self.train_step(batch, criterion)

    def eval_log(self, ap, batch, outputs):
        return self.train_log(ap, batch, outputs)
