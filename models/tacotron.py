# coding: utf-8
import torch
from torch import nn
from math import sqrt
from utils.text.symbols import symbols, phonemes
from layers.tacotron import Prenet, Encoder, Decoder, PostCBHG


class Tacotron(nn.Module):
    def __init__(self,
                 embedding_dim=256,
                 linear_dim=1025,
                 mel_dim=80,
                 r=5,
                 padding_idx=None):
        super(Tacotron, self).__init__()
        self.r = r
        self.mel_dim = mel_dim
        self.linear_dim = linear_dim
        self.embedding = nn.Embedding(
            len(phonemes), embedding_dim, padding_idx=padding_idx)
        print(" | > Number of characters : {}".format(len(phonemes)))
        std = sqrt(2.0 / (len(phonemes) + embedding_dim))
        val = sqrt(3.0) * std  # uniform bounds for std
        self.embedding.weight.data.uniform_(-val, val)
        self.encoder = Encoder(embedding_dim)
        self.decoder = Decoder(256, mel_dim, r)
        self.postnet = PostCBHG(mel_dim)
        self.last_linear = nn.Sequential(
            nn.Linear(self.postnet.cbhg.gru_features * 2, linear_dim),
            nn.Sigmoid())

    def forward(self, characters, mel_specs=None, mask=None):
        B = characters.size(0)
        inputs = self.embedding(characters)
        # batch x time x dim
        encoder_outputs = self.encoder(inputs)
        # batch x time x dim*r
        mel_outputs, alignments, stop_tokens = self.decoder(
            encoder_outputs, mel_specs, mask)
        # Reshape
        # batch x time x dim
        mel_outputs = mel_outputs.view(B, -1, self.mel_dim)
        linear_outputs = self.postnet(mel_outputs)
        linear_outputs = self.last_linear(linear_outputs)
        return mel_outputs, linear_outputs, alignments, stop_tokens
