# coding: utf-8
import torch
from torch import nn

from TTS.tts.layers.tacotron.gst_layers import GST
from TTS.tts.layers.tacotron.tacotron import Decoder, Encoder, PostCBHG
from TTS.tts.models.tacotron_abstract import TacotronAbstract


class Tacotron(TacotronAbstract):
    """Tacotron as in https://arxiv.org/abs/1703.10135

    It's an autoregressive encoder-attention-decoder-postnet architecture.

    Args:
        num_chars (int): number of input characters to define the size of embedding layer.
        num_speakers (int): number of speakers in the dataset. >1 enables multi-speaker training and model learns speaker embeddings.
        r (int): initial model reduction rate.
        postnet_output_dim (int, optional): postnet output channels. Defaults to 80.
        decoder_output_dim (int, optional): decoder output channels. Defaults to 80.
        attn_type (str, optional): attention type. Check ```TTS.tts.layers.attentions.init_attn```. Defaults to 'original'.
        attn_win (bool, optional): enable/disable attention windowing.
            It especially useful at inference to keep attention alignment diagonal. Defaults to False.
        attn_norm (str, optional): Attention normalization method. "sigmoid" or "softmax". Defaults to "softmax".
        prenet_type (str, optional): prenet type for the decoder. Defaults to "original".
        prenet_dropout (bool, optional): prenet dropout rate. Defaults to True.
        forward_attn (bool, optional): enable/disable forward attention.
            It is only valid if ```attn_type``` is ```original```.  Defaults to False.
        trans_agent (bool, optional): enable/disable transition agent in forward attention. Defaults to False.
        forward_attn_mask (bool, optional): enable/disable extra masking over forward attention. Defaults to False.
        location_attn (bool, optional): enable/disable location sensitive attention.
            It is only valid if ```attn_type``` is ```original```. Defaults to True.
        attn_K (int, optional): Number of attention heads for GMM attention. Defaults to 5.
        separate_stopnet (bool, optional): enable/disable separate stopnet training without only gradient
            flow from stopnet to the rest of the model.  Defaults to True.
        bidirectional_decoder (bool, optional): enable/disable bidirectional decoding. Defaults to False.
        double_decoder_consistency (bool, optional): enable/disable double decoder consistency. Defaults to False.
        ddc_r (int, optional): reduction rate for the coarse decoder of double decoder consistency. Defaults to None.
        encoder_in_features (int, optional): input channels for the encoder. Defaults to 512.
        decoder_in_features (int, optional): input channels for the decoder. Defaults to 512.
        speaker_embedding_dim (int, optional): external speaker conditioning vector channels. Defaults to None.
        gst (bool, optional): enable/disable global style token learning. Defaults to False.
        gst_embedding_dim (int, optional): size of channels for GST vectors. Defaults to 512.
        gst_num_heads (int, optional): number of attention heads for GST. Defaults to 4.
        gst_style_tokens (int, optional): number of GST tokens. Defaults to 10.
        gst_use_speaker_embedding (bool, optional): enable/disable inputing speaker embedding to GST. Defaults to False.
        memory_size (int, optional): size of the history queue fed to the prenet. Model feeds the last ```memory_size```
            output frames to the prenet.
    """
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
                 encoder_in_features=256,
                 decoder_in_features=256,
                 speaker_embedding_dim=None,
                 gst=False,
                 gst_embedding_dim=256,
                 gst_num_heads=4,
                 gst_style_tokens=10,
                 memory_size=5,
                 gst_use_speaker_embedding=False):
        super(Tacotron,
              self).__init__(num_chars, num_speakers, r, postnet_output_dim,
                             decoder_output_dim, attn_type, attn_win,
                             attn_norm, prenet_type, prenet_dropout,
                             forward_attn, trans_agent, forward_attn_mask,
                             location_attn, attn_K, separate_stopnet,
                             bidirectional_decoder, double_decoder_consistency,
                             ddc_r, encoder_in_features, decoder_in_features,
                             speaker_embedding_dim, gst, gst_embedding_dim,
                             gst_num_heads, gst_style_tokens, gst_use_speaker_embedding)

        # speaker embedding layers
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                speaker_embedding_dim = 256
                self.speaker_embedding = nn.Embedding(self.num_speakers, speaker_embedding_dim)
                self.speaker_embedding.weight.data.normal_(0, 0.3)

        # speaker and gst embeddings is concat in decoder input
        if self.num_speakers > 1:
            self.decoder_in_features += speaker_embedding_dim # add speaker embedding dim

        # embedding layer
        self.embedding = nn.Embedding(num_chars, 256, padding_idx=0)
        self.embedding.weight.data.normal_(0, 0.3)

        # base model layers
        self.encoder = Encoder(self.encoder_in_features)
        self.decoder = Decoder(self.decoder_in_features, decoder_output_dim, r,
                               memory_size, attn_type, attn_win, attn_norm,
                               prenet_type, prenet_dropout, forward_attn,
                               trans_agent, forward_attn_mask, location_attn,
                               attn_K, separate_stopnet)
        self.postnet = PostCBHG(decoder_output_dim)
        self.last_linear = nn.Linear(self.postnet.cbhg.gru_features * 2,
                                     postnet_output_dim)

        # global style token layers
        if self.gst:
            self.gst_layer = GST(num_mel=80,
                                 num_heads=gst_num_heads,
                                 num_style_tokens=gst_style_tokens,
                                 gst_embedding_dim=self.gst_embedding_dim,
                                 speaker_embedding_dim=speaker_embedding_dim if self.embeddings_per_sample and self.gst_use_speaker_embedding else None)
        # backward pass decoder
        if self.bidirectional_decoder:
            self._init_backward_decoder()
        # setup DDC
        if self.double_decoder_consistency:
            self.coarse_decoder = Decoder(
                self.decoder_in_features, decoder_output_dim, ddc_r, memory_size,
                attn_type, attn_win, attn_norm, prenet_type, prenet_dropout,
                forward_attn, trans_agent, forward_attn_mask, location_attn,
                attn_K, separate_stopnet)

    def forward(self, characters, text_lengths, mel_specs, mel_lengths=None, speaker_ids=None, speaker_embeddings=None):
        """
        Shapes:
            characters: [B, T_in]
            text_lengths: [B]
            mel_specs: [B, T_out, C]
            mel_lengths: [B]
            speaker_ids: [B, 1]
            speaker_embeddings: [B, C]
        """
        input_mask, output_mask = self.compute_masks(text_lengths, mel_lengths)
        # B x T_in x embed_dim
        inputs = self.embedding(characters)
        # B x T_in x encoder_in_features
        encoder_outputs = self.encoder(inputs)
        # sequence masking
        encoder_outputs = encoder_outputs * input_mask.unsqueeze(2).expand_as(encoder_outputs)
        # global style token
        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs,
                                               mel_specs,
                                               speaker_embeddings if self.gst_use_speaker_embedding else None)
        # speaker embedding
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            else:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = torch.unsqueeze(speaker_embeddings, 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)
        # decoder_outputs: B x decoder_in_features x T_out
        # alignments: B x T_in x encoder_in_features
        # stop_tokens: B x T_in
        decoder_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, input_mask)
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
    def inference(self, characters, speaker_ids=None, style_mel=None, speaker_embeddings=None):
        inputs = self.embedding(characters)
        encoder_outputs = self.encoder(inputs)
        if self.gst:
            # B x gst_dim
            encoder_outputs = self.compute_gst(encoder_outputs,
                                               style_mel,
                                               speaker_embeddings if self.gst_use_speaker_embedding else None)
        if self.num_speakers > 1:
            if not self.embeddings_per_sample:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = self.speaker_embedding(speaker_ids)[:, None]
            else:
                # B x 1 x speaker_embed_dim
                speaker_embeddings = torch.unsqueeze(speaker_embeddings, 1)
            encoder_outputs = self._concat_speaker_embedding(encoder_outputs, speaker_embeddings)
        decoder_outputs, alignments, stop_tokens = self.decoder.inference(
            encoder_outputs)
        postnet_outputs = self.postnet(decoder_outputs)
        postnet_outputs = self.last_linear(postnet_outputs)
        decoder_outputs = decoder_outputs.transpose(1, 2)
        return decoder_outputs, postnet_outputs, alignments, stop_tokens
