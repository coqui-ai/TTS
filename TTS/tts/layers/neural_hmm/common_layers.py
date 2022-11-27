import torch.nn as nn

from TTS.tts.layers.tacotron.tacotron2 import ConvBNBlock


class Encoder(nn.Module):
    r"""Neural HMM Encoder
    
    Same as Tacotron 2 encoder but increases the states per phone

    Args:
        in_out_channels (int): number of input and output channels.

    Shapes:
        - input: (B, C_in, T)
        - output: (B, C_in, T)
    """

    def __init__(self, state_per_phone, in_out_channels=512):
        super().__init__()

        self.state_per_phone = state_per_phone
        self.in_out_channels = in_out_channels
        
        
        self.convolutions = nn.ModuleList()
        for _ in range(3):
            self.convolutions.append(ConvBNBlock(in_out_channels, in_out_channels, 5, "relu"))
        self.lstm = nn.LSTM(
            in_out_channels,
            int(in_out_channels / 2) * state_per_phone,
            num_layers=1,
            batch_first=True,
            bias=True,
            bidirectional=True
        )
        self.rnn_state = None
        

    def forward(self, x, input_lengths):
        b, _, T = x.shape
        o = x
        for layer in self.convolutions:
            o = layer(o)
        o = o.transpose(1, 2)
        o = nn.utils.rnn.pack_padded_sequence(o, input_lengths.cpu(), batch_first=True)
        self.lstm.flatten_parameters()
        o, _ = self.lstm(o)
        o, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True)
        o = o.reshape(b, T * self.state_per_phone, self.in_out_channels)
        T = input_lengths * self.state_per_phone
        return o, T


