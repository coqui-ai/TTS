import torch
import torch.nn as nn
import torch.nn.functional as F
from TTS.tts.utils.generic_utils import extract_axis_1

class CapacitronVAE(nn.Module):
    """Effective Use of Variational Embedding Capacity for prosody transfer.

    See https://arxiv.org/abs/1906.03402 """

    def __init__(self, num_mel, capacitron_embedding_dim, speaker_embedding_dim=None, text_summary_embedding_dim=None):
        super().__init__()
        self.encoder = ReferenceEncoder(num_mel, capacitron_embedding_dim)
        # TODO: Figure out what to do with speaker_embedding_dim

        if text_summary_embedding_dim is not None:
            self.text_summary_net = TextSummary(text_summary_embedding_dim)

    def forward(self, inputs, mel_lengths, text_info=None, speaker_embedding=None):
        enc_out = self.encoder(inputs, mel_lengths)
        # concat speaker_embedding and/or text summary embedding
        if text_info is not None:
            text_inputs = text_info[0]
            input_lengths = text_info[1]
            text_summary_out = self.text_summary_net(text_inputs, input_lengths).cuda()
            enc_out = torch.cat([enc_out, text_summary_out], dim=-1)
        if speaker_embedding is not None:
            enc_out = torch.cat([enc_out, speaker_embedding], dim=-1)
        # reshape to [batch_size, 1, speaker_embed_dim + text_embed_dim]
        return torch.unsqueeze(enc_out, 1)


class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1)) for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features=filter_size)
            for filter_size in filters[1:]
        ])

        post_conv_height = self.calculate_post_conv_height(
            num_mel, 3, 2, 1, num_layers)
        # TODO post_conv_lenght also needs to be calculated
        self.recurrence = nn.LSTM(
            input_size=filters[-1] * post_conv_height,
            hidden_size=embedding_dim,
            batch_first=True,
            bidirectional=False)

    def forward(self, inputs, input_lengths):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        # Get last value of LSTM
        post_conv_input_lengths = [post_conv_width] * batch_size
        o, _ = self.recurrence(x)
        last_output = extract_axis_1(o.cpu(), torch.LongTensor(post_conv_input_lengths).cpu())
        return last_output.cuda()

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad,
                                   n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height

class TextSummary(nn.Module):
    def __init__(self, embedding_dim):
        super(TextSummary, self).__init__()
        self.lstm = nn.LSTM(256, # 256 is the hardcoded text embedding dimension from the text encoder
                            embedding_dim, # fixed length output summary the lstm creates from the input
                            batch_first=True,
                            bidirectional=False)

    def forward(self, inputs, input_lengths):
        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        packed_seqs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.tolist(), batch_first=True, enforce_sorted=False) # dynamic rnn sequence padding
        o, _ = self.lstm(packed_seqs)
        out_dynamic, _ = nn.utils.rnn.pad_packed_sequence(o, batch_first=True) # inverse repadding
        last_output = extract_axis_1(out_dynamic.cpu(), input_lengths.cpu())
        return last_output
