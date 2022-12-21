import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class RangePredictor(nn.Module):
    """
    Range Predictor module as in https://arxiv.org/pdf/2010.04301.pdf

    Model::
        x -> 2 x BiLSTM -> Linear -> o
    """

    def __init__(self, in_channels, hidden_channels):
        super(RangePredictor, self).__init__()
        assert hidden_channels % 2 == 0, "range_lstm_dim must be even [{}]".format(in_channels)

        self.lstm = nn.LSTM(in_channels + 1, int(hidden_channels / 2), 2, batch_first=True, bidirectional=True)

        self.proj = nn.Linear(hidden_channels, 1, bias=True)

    def forward(self, x, dr, x_lengths=None):
        x = x.transpose(1, 2)  # [B, C, T] -> [B, T, C]
        concated_inputs = torch.cat([x, dr.unsqueeze(-1)], dim=-1)

        ## remove pad activations
        if x_lengths is not None:
            concated_inputs = pack_padded_sequence(
                concated_inputs, x_lengths.cpu().int(), batch_first=True, enforce_sorted=False
            )

        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(concated_inputs)

        if x_lengths is not None:
            outputs, _ = pad_packed_sequence(outputs, batch_first=True)

        outputs = self.proj(outputs)
        outputs = F.softplus(outputs)
        return outputs.squeeze(1)
        