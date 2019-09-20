# coding: utf-8
import torch
from torch import nn
from TTS.layers.tacotron import Encoder, Decoder, PostCBHG
from TTS.layers.gst_layers import GST
from TTS.utils.generic_utils import sequence_mask
from TTS.models.tacotron import Tacotron


class TacotronGST(Tacotron):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r=5,
                 linear_dim=1025,
                 mel_dim=80,
                 memory_size=5,
                 attn_win=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 separate_stopnet=True):
        super().__init__(num_chars,
                        num_speakers,
                        r,
                        linear_dim,
                        mel_dim,
                        memory_size,
                        attn_win,
                        attn_norm,
                        prenet_type,
                        prenet_dropout,
                        forward_attn,
                        trans_agent,
                        forward_attn_mask,
                        location_attn,
                        separate_stopnet)
        gst_embedding_dim = 256
        decoder_dim = 512 + gst_embedding_dim if num_speakers > 1 else 256 + gst_embedding_dim
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        self.decoder = Decoder(decoder_dim, mel_dim, r, memory_size, attn_win,
                               attn_norm, prenet_type, prenet_dropout,
                               forward_attn, trans_agent, forward_attn_mask,
                               location_attn, separate_stopnet, proj_speaker_dim)
        self.gst = GST(num_mel=80, num_heads=4,
                       num_style_tokens=10, embedding_dim=gst_embedding_dim)

    def forward(self, characters, text_lengths, mel_specs, speaker_ids=None):
        B = characters.size(0)
        mask = sequence_mask(text_lengths).to(characters.device)
        inputs = self.embedding(characters)
        self._init_states()
        self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            inputs = self._add_speaker_embedding(inputs,
                                                self.speaker_embeddings)
        encoder_outputs = self.encoder(inputs)
        if self.num_speakers > 1:
            encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                        self.speaker_embeddings)
        gst_outputs = self.gst(mel_specs)
        encoder_outputs = self._add_speaker_embedding(
            encoder_outputs, gst_outputs)
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask, self.speaker_embeddings_projected)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens

    def inference(self, characters, speaker_ids=None, style_mel=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        self._init_states()
        self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            inputs = self._add_speaker_embedding(inputs,
                                                self.speaker_embeddings)
        encoder_outputs = self.encoder(inputs)
        if self.num_speakers > 1:
            encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                        self.speaker_embeddings)
        if style_mel is not None:
            gst_outputs = self.gst(style_mel)
            gst_outputs = gst_outputs.expand(-1, encoder_outputs.size(1), -1)
            encoder_outputs = self._add_speaker_embedding(encoder_outputs,
                                                             gst_outputs)
        mel_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs, self.speaker_embeddings_projected)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens
