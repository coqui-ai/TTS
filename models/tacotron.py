# coding: utf-8
import torch
from torch.autograd import Variable
from torch import nn
from TTS.utils.text.symbols import symbols
from TTS.layers.tacotron import Prenet, Encoder, Decoder, CBHG


class Tacotron(nn.Module):
    def __init__(self, embedding_dim=256, linear_dim=1025, mel_dim=80,
                 freq_dim=1025, r=5, padding_idx=None):
                 
        super(Tacotron, self).__init__()
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(len(symbols), embedding_dim,
                                      padding_idx=padding_idx)
        print(" | > Embedding dim : {}".format(len(symbols)))

        # Trying smaller std
        self.embedding.weight.data.normal_(0, 0.3)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(256, mel_dim, r)

        self.postnet = CBHG(mel_dim, K=8, projections=[256, mel_dim])
        self.last_linear = nn.Linear(mel_dim * 2, freq_dim)

    def forward(self, characters, mel_specs=None):
        B = characters.size(0)

        inputs = self.embedding(characters)
        # (B, T', in_dim)
        encoder_outputs = self.encoder(inputs)

        # (B, T', mel_dim*r)
        mel_outputs, alignments = self.decoder(
            encoder_outputs, mel_specs)

        # Post net processing below

        # Reshape
        # (B, T, mel_dim)
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)

        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)

        return mel_outputs, linear_outputs, alignments
