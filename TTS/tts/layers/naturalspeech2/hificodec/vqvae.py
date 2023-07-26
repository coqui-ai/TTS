import json

import torch
import torch.nn as nn

from .env import AttrDict
from TTS.tts.layers.naturalspeech2.hificodec.models import Encoder
from TTS.tts.layers.naturalspeech2.hificodec.models import Generator
from TTS.tts.layers.naturalspeech2.hificodec.models import Quantizer


class VQVAE(nn.Module):
    def __init__(self,
                 config_path,
                 ckpt_path,
                 with_encoder=False):
        super(VQVAE, self).__init__()
        ckpt = torch.load(ckpt_path)
        with open(config_path) as f:
            data = f.read()
        json_config = json.loads(data)
        self.h = AttrDict(json_config)
        self.quantizer = Quantizer(self.h)
        self.generator = Generator(self.h)
        self.generator.load_state_dict(ckpt['generator'])
        self.quantizer.load_state_dict(ckpt['quantizer'])
        if with_encoder:
            self.encoder = Encoder(self.h)
            self.encoder.load_state_dict(ckpt['encoder'])

    def forward(self, x):
        # x is the codebook
        # x.shape (B, T, Nq)
        quant_emb = self.quantizer.embed(x)
        return self.generator(quant_emb)

    def encode(self, x):
        batch_size = x.size(0)
        if len(x.shape) == 3 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        c = self.encoder(x)
        q, loss_q, c = self.quantizer(c)
        c = [code.reshape(batch_size, -1) for code in c]
        # shape: [N, T, 4]
        return torch.stack(c, -1)
