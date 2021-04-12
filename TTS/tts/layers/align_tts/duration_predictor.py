from torch import nn

from TTS.tts.layers.generic.pos_encoding import PositionalEncoding
from TTS.tts.layers.generic.transformer import FFTransformerBlock


class DurationPredictor(nn.Module):
    def __init__(self, num_chars, hidden_channels, hidden_channels_ffn, num_heads):
        super().__init__()
        self.embed = nn.Embedding(num_chars, hidden_channels)
        self.pos_enc = PositionalEncoding(hidden_channels, dropout_p=0.1)
        self.FFT = FFTransformerBlock(hidden_channels, num_heads, hidden_channels_ffn, 2, 0.1)
        self.out_layer = nn.Conv1d(hidden_channels, 1, 1)

    def forward(self, text, text_lengths):
        # B, L -> B, L
        emb = self.embed(text)
        emb = self.pos_enc(emb.transpose(1, 2))
        x = self.FFT(emb, text_lengths)
        x = self.out_layer(x).squeeze(-1)
        return x
