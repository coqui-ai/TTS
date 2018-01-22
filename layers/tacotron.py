# coding: utf-8
import torch
from torch.autograd import Variable
from torch import nn

from .attention import BahdanauAttention, AttentionWrapper
from .attention import get_mask_from_lengths

class Prenet(nn.Module):
    def __init__(self, in_dim, sizes=[256, 128]):
        super(Prenet, self).__init__()
        in_sizes = [in_dim] + sizes[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_sizes, sizes)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))

        return inputs


class BatchNormConv1d(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_dim, out_dim,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_dim, momentum=0.99, eps=1e-3)
        self.activation = activation

    def forward(self, x):
        x = self.conv1d(x)
        if self.activation is not None:
            x = self.activation(x)
        return self.bn(x)


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        H = self.relu(self.H(inputs))
        T = self.sigmoid(self.T(inputs))
        return H * T + inputs * (1.0 - T)


class CBHG(nn.Module):
    """CBHG module: a recurrent neural network composed of:
        - 1-d convolution banks
        - Highway networks + residual connections
        - Bidirectional gated recurrent units
    """

    def __init__(self, in_dim, K=16, projections=[128, 128]):
        super(CBHG, self).__init__()
        self.in_dim = in_dim
        self.relu = nn.ReLU()
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_dim, in_dim, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
             for k in range(1, K + 1)])
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)

        in_sizes = [K * in_dim] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1) + [None]
        self.conv1d_projections = nn.ModuleList(
            [BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                             padding=1, activation=ac)
             for (in_size, out_size, ac) in zip(
                 in_sizes, projections, activations)])

        self.pre_highway = nn.Linear(projections[-1], in_dim, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_dim, in_dim) for _ in range(4)])

        self.gru = nn.GRU(
            in_dim, in_dim, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs, input_lengths=None):
        # (B, T_in, in_dim)
        x = inputs

        # Needed to perform conv1d on time-axis
        # (B, in_dim, T_in)
        if x.size(-1) == self.in_dim:
            x = x.transpose(1, 2)

        T = x.size(-1)

        # (B, in_dim*K, T_in)
        # Concat conv1d bank outputs
        x = torch.cat([conv1d(x)[:, :, :T] for conv1d in self.conv1d_banks], dim=1)
        assert x.size(1) == self.in_dim * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]

        for conv1d in self.conv1d_projections:
            x = conv1d(x)

        # (B, T_in, in_dim)
        # Back to the original shape
        x = x.transpose(1, 2)

        if x.size(-1) != self.in_dim:
            x = self.pre_highway(x)

        # Residual connection
        x += inputs
        for highway in self.highways:
            x = highway(x)

        if input_lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, input_lengths, batch_first=True)

        # (B, T_in, in_dim*2)
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)

        if input_lengths is not None:
            outputs, _ = nn.utils.rnn.pad_packed_sequence(
                outputs, batch_first=True)

        return outputs


class Encoder(nn.Module):
    def __init__(self, in_dim):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_dim, sizes=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs, input_lengths=None):
        inputs = self.prenet(inputs)
        return self.cbhg(inputs, input_lengths)


class Decoder(nn.Module):
    def __init__(self, memory_dim, r):
        super(Decoder, self).__init__()
        self.memory_dim = memory_dim
        self.r = r
        self.prenet = Prenet(memory_dim * r, sizes=[256, 128])
        # attetion RNN
        self.attention_rnn = AttentionWrapper(
            nn.GRUCell(256 + 128, 256),
            BahdanauAttention(256)
        )
        
        self.memory_layer = nn.Linear(256, 256, bias=False)

        # concat and project context and attention vectors
        # (prenet_out + attention context) -> output
        self.project_to_decoder_in = nn.Linear(512, 256)

        # decoder RNNs
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])

        self.proj_to_mel = nn.Linear(256, memory_dim * r)
        self.max_decoder_steps = 200

    def forward(self, decoder_inputs, memory=None, memory_lengths=None):
        """
        Decoder forward step.

        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.

        Args:
            decoder_inputs: Encoder outputs. (B, T_encoder, dim)
            memory: Decoder memory. i.e., mel-spectrogram. If None (at eval-time),
              decoder outputs are used as decoder inputs.
            memory_lengths: Encoder output (memory) lengths. If not None, used for
              attention masking.
        """
        B = decoder_inputs.size(0)

        processed_memory = self.memory_layer(decoder_inputs)
        if memory_lengths is not None:
            mask = get_mask_from_lengths(processed_memory, memory_lengths)
        else:
            mask = None

        # Run greedy decoding if memory is None
        greedy = memory is None

        if memory is not None:
            # Grouping multiple frames if necessary
            if memory.size(-1) == self.memory_dim:
                memory = memory.view(B, memory.size(1) // self.r, -1)
            assert memory.size(-1) == self.memory_dim * self.r,\
                " !! Dimension mismatch {} vs {} * {}".format(memory.size(-1),
                                                         self.memory_dim, self.r)
            T_decoder = memory.size(1)

        # go frames - 0 frames tarting the sequence
        initial_input = Variable(
            decoder_inputs.data.new(B, self.memory_dim * self.r).zero_())

        # Init decoder states
        attention_rnn_hidden = Variable(
            decoder_inputs.data.new(B, 256).zero_())
        decoder_rnn_hiddens = [Variable(
            decoder_inputs.data.new(B, 256).zero_())
            for _ in range(len(self.decoder_rnns))]
        current_attention = Variable(
            decoder_inputs.data.new(B, 256).zero_())

        # Time first (T_decoder, B, memory_dim)
        if memory is not None:
            memory = memory.transpose(0, 1)

        outputs = []
        alignments = []

        t = 0
        current_input = initial_input
        while True:
            if t > 0:
                current_input = outputs[-1] if greedy else memory[t - 1]
            # Prenet
            current_input = self.prenet(current_input)

            # Attention RNN
            attention_rnn_hidden, current_attention, alignment = self.attention_rnn(
                current_input, current_attention, attention_rnn_hidden,
                decoder_inputs, processed_memory=processed_memory, mask=mask)

            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_attention), -1))

            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input

            output = decoder_input

            # predict mel vectors from decoder vectors
            output = self.proj_to_mel(output)

            outputs += [output]
            alignments += [alignment]

            t += 1

            if greedy:
                if t > 1 and is_end_of_frames(output):
                    break
                elif t > self.max_decoder_steps:
                    print("Warning! doesn't seems to be converged")
                    break
            else:
                if t >= T_decoder:
                    break

        assert greedy or len(outputs) == T_decoder

        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()

        return outputs, alignments


def is_end_of_frames(output, eps=0.2):
    return (output.data <= eps).all()
