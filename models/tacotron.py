# coding: utf-8
import torch
from torch import nn

from TTS.layers.gst_layers import GST
from TTS.layers.tacotron import Decoder, Encoder, PostCBHG
from TTS.models.tacotron_abstract import TacotronAbstract


class Tacotron(TacotronAbstract):
    def __init__(self,
                 num_chars,
                 num_speakers,
                 r=5,
                 postnet_output_dim=1025,
                 decoder_output_dim=80,
                 attn_type='original',
                 attn_win=False,
                 attn_norm="sigmoid",
                 prenet_type="original",
                 prenet_dropout=True,
                 forward_attn=False,
                 trans_agent=False,
                 forward_attn_mask=False,
                 location_attn=True,
                 attn_K=5,
                 separate_stopnet=True,
                 bidirectional_decoder=False,
                 double_decoder_consistency=False,
                 ddc_r=None,
                 gst=False,
                 memory_size=5):
        super(Tacotron,
              self).__init__(num_chars, num_speakers, r, postnet_output_dim,
                             decoder_output_dim, attn_type, attn_win,
                             attn_norm, prenet_type, prenet_dropout,
                             forward_attn, trans_agent, forward_attn_mask,
                             location_attn, attn_K, separate_stopnet,
                             bidirectional_decoder, double_decoder_consistency,
                             ddc_r, gst)
        decoder_in_features = 512 if num_speakers > 1 else 256
        encoder_in_features = 512 if num_speakers > 1 else 256
        speaker_embedding_dim = 256
        proj_speaker_dim = 80 if num_speakers > 1 else 0
        # base model layers
        self.embedding = nn.Embedding(num_chars, 256, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(encoder_in_features)
        self.decoder = Decoder(decoder_in_features, decoder_output_dim, r,
                               memory_size, attn_type, attn_win, attn_norm,
                               prenet_type, prenet_dropout, forward_attn,
                               trans_agent, forward_attn_mask, location_attn,
                               attn_K, separate_stopnet, proj_speaker_dim)
        self.postnet = PostCBHG(decoder_output_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2,
                                     postnet_output_dim)
        # speaker embedding layers
        if num_speakers > 1:
            self.speaker_embedding = nn.Embedding(num_speakers, speaker_embedding_dim)
            self.speaker_embedding.weight.data.normal_(0, 0.3)
            self.speaker_project_mel = nn.Sequential(
                nn.Linear(speaker_embedding_dim, proj_speaker_dim), nn.Tanh())
            self.speaker_embeddings = None
            self.speaker_embeddings_projected = None
        # global style token layers
        if self.gst:
            gst_embedding_dim = 256
            self.gst_layer = GST(num_mel=80,
                                 num_heads=4,
                                 num_style_tokens=10,
                                 embedding_dim=gst_embedding_dim)
        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                decoder_in_features, decoder_output_dim, ddc_r, memory_size,
                attn_type, attn_win, attn_norm, prenet_type, prenet_dropout,
                forward_attn, trans_agent, forward_attn_mask, location_attn,
                attn_K, separate_stopnet, proj_speaker_dim)


    def forward(self, characters, text_lengths, mel_specs, mel_lengths=None, speaker_ids=None):
        """
        Shapes:
            - characters: B x T_in
            - text_lengths: B
            - mel_specs: B x T_out x D
            - speaker_ids: B x 1
        """
        self._init_states()
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x T_in x embed_dim
        inputs = self.embedding(characters)
        # B x speaker_embed_dim
        if speaker_ids is not None:
            self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            # B x T_in x embed_dim + speaker_embed_dim
            inputs = self._concat_speaker_embedding(inputs,
                                                    self.speaker_embeddings)
        # B x T_in x encoder_in_features
        encoder_outputs = self.encoder(inputs)
        # sequence masking
        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)
        # global style token
        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs, mel_specs)
        if self.num_speakers > 1:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs, self.speaker_embeddings)
        # decoder_outputs: B x decoder_in_features x T_out
        # alignments: B x T_in x encoder_in_features
        # stop_tokens: B x T_in
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, input_mask,
            self.speaker_embeddings_projected)
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
            return decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        if self.double_decoder_consistency:
            decoder_outputs_backward, alignments_backward = self._coarse_decoder_pass(mel_specs, encoder_outputs, alignments, input_mask)
            return  decoder_outputs, postnet_outputs, alignments, stop_tokens, decoder_outputs_backward, alignments_backward
        return decoder_outputs, postnet_outputs, alignments, stop_tokens

    @torch.no_grad()
    def inference(self, characters, speaker_ids=None, style_mel=None):
        inputs = self.embedding(characters)
        self._init_states()
        if speaker_ids is not None:
            self.compute_speaker_embedding(speaker_ids)
        if self.num_speakers > 1:
            inputs = self._concat_speaker_embedding(inputs,
                                                    self.speaker_embeddings)
        encoder_outputs = self.encoder(inputs)
        if self.gst and style_mel is not None:
            encoder_outputs = self.compute_gst(encoder_outputs, style_mel)
        if self.num_speakers > 1:
            encoder_outputs = self._concat_speaker_embedding(
                encoder_outputs, self.speaker_embeddings)
        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs, self.speaker_embeddings_projected)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = self.last_linear(postnet_outputs)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens
