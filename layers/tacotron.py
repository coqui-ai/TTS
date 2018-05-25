# coding: utf-8
import torch
from torch import nn
from .attention import AttentionRNN
from .attention import get_mask_from_lengths


class Prenet(nn.Module):
    r""" Prenet as explained at https://arxiv.org/abs/1703.10135.
    It creates as many layers as given by 'out_features'

    Args:
        in_features (int): size of the input vector
        out_features (int or list): size of each output sample.
            If it is a list, for each value, there is created a new layer.  
    """

    def __init__(self, in_features, out_features=[256, 128]):
        super(Prenet, self).__init__()
        in_features = [in_features] + out_features[:-1]
        self.layers = nn.ModuleList(
            [nn.Linear(in_size, out_size)
             for (in_size, out_size) in zip(in_features, out_features)])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, inputs):
        for linear in self.layers:
            inputs = self.dropout(self.relu(linear(inputs)))
        return inputs


class BatchNormConv1d(nn.Module):
    r"""A wrapper for Conv1d with BatchNorm. It sets the activation
    function between Conv and BatchNorm layers. BatchNorm layer
    is initialized with the TF default values for momentum and eps.

    Args:
        in_channels: size of each input sample
        out_channels: size of each output samples
        kernel_size: kernel size of conv filters
        stride: stride of conv filters
        padding: padding of conv filters
        activation: activation function set b/w Conv1d and BatchNorm

    Shapes:
        - input: batch x dims
        - output: batch x dims
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 activation=None):
        super(BatchNormConv1d, self).__init__()
        self.conv1d = nn.Conv1d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride, padding=padding, bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
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

        Args:
            in_features (int): sample size
            K (int): max filter size in conv bank
            projections (list): conv channel sizes for conv projections
            num_highways (int): number of highways layers

        Shapes:
            - input: batch x time x dim
            - output: batch x time x dim*2
    """

    def __init__(self, in_features, K=16, projections=[128, 128], num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.relu = nn.ReLU()
        # list of conv1d bank with filter size k=1...K
        # TODO: try dilational layers instead
        self.conv1d_banks = nn.ModuleList(
            [BatchNormConv1d(in_features, in_features, kernel_size=k, stride=1,
                             padding=k // 2, activation=self.relu)
                for k in range(1, K + 1)])
        # max pooling of conv bank
        # TODO: try average pooling OR larger kernel size
        self.max_pool1d = nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
        out_features = [K * in_features] + projections[:-1]
        activations = [self.relu] * (len(projections) - 1)
        activations += [None]
        # setup conv1d projection layers
        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, projections, activations):
            layer = BatchNormConv1d(in_size, out_size, kernel_size=3, stride=1,
                                    padding=1, activation=ac)
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)
        # setup Highway layers
        self.pre_highway = nn.Linear(projections[-1], in_features, bias=False)
        self.highways = nn.ModuleList(
            [Highway(in_features, in_features) for _ in range(num_highways)])
        # bi-directional GPU layer
        self.gru = nn.GRU(
            in_features, in_features, 1, batch_first=True, bidirectional=True)

    def forward(self, inputs):
        # (B, T_in, in_features)
        x = inputs
        # Needed to perform conv1d on time-axis
        # (B, in_features, T_in)
        if x.size(-1) == self.in_features:
            x = x.transpose(1, 2)
        T = x.size(-1)
        # (B, in_features*K, T_in)
        # Concat conv1d bank outputs
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            out = out[:, :, :T]
            outs.append(out)
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.in_features * len(self.conv1d_banks)
        x = self.max_pool1d(x)[:, :, :T]
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        # (B, T_in, in_features)
        # Back to the original shape
        x = x.transpose(1, 2)
        if x.size(-1) != self.in_features:
            x = self.pre_highway(x)
        # Residual connection
        # TODO: try residual scaling as in Deep Voice 3
        # TODO: try plain residual layers
        x += inputs
        for highway in self.highways:
            x = highway(x)
        # (B, T_in, in_features*2)
        # TODO: replace GRU with convolution as in Deep Voice 3
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs


class Encoder(nn.Module):
    r"""Encapsulate Prenet and CBHG modules for encoder"""

    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_features, out_features=[256, 128])
        self.cbhg = CBHG(128, K=16, projections=[128, 128])

    def forward(self, inputs):
        r"""
        Args:
            inputs (FloatTensor): embedding features

        Shapes:
            - inputs: batch x time x in_features
            - outputs: batch x time x 128*2
        """
        inputs = self.prenet(inputs)
        return self.cbhg(inputs)


class Decoder(nn.Module):
    r"""Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
        eps (float): threshold for detecting the end of a sentence.
    """

    def __init__(self, in_features, memory_dim, r):
        super(Decoder, self).__init__()
        self.r = r
        self.max_decoder_steps = 200
        self.memory_dim = memory_dim
        # memory -> |Prenet| -> processed_memory
        self.prenet = Prenet(memory_dim * r, out_features=[256, 128])
        # processed_inputs, processed_memory -> |Attention| -> Attention, Alignment, RNN_State
        self.attention_rnn = AttentionRNN(256, in_features, 128)
        # (processed_memory | attention context) -> |Linear| -> decoder_RNN_input
        self.project_to_decoder_in = nn.Linear(256+in_features, 256)
        # decoder_RNN_input -> |RNN| -> RNN_state
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(256, memory_dim * r)
        self.stopnet = StopNet(r, memory_dim)

    def forward(self, inputs, memory=None):
        """
        Decoder forward step.

        If decoder inputs are not given (e.g., at testing time), as noted in
        Tacotron paper, greedy decoding is adapted.

        Args:
            inputs: Encoder outputs.
            memory (None): Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        B = inputs.size(0)
        # Run greedy decoding if memory is None
        greedy = not self.training
        if memory is not None:
            # Grouping multiple frames if necessary
            if memory.size(-1) == self.memory_dim:
                memory = memory.view(B, memory.size(1) // self.r, -1)
                " !! Dimension mismatch {} vs {} * {}".format(memory.size(-1),
                                                              self.memory_dim, self.r)
            T_decoder = memory.size(1)
        # go frame - 0 frames tarting the sequence
        initial_memory = inputs.data.new(B, self.memory_dim * self.r).zero_()
        # Init decoder states
        attention_rnn_hidden = inputs.data.new(B, 256).zero_()
        decoder_rnn_hiddens = [inputs.data.new(B, 256).zero_()
            for _ in range(len(self.decoder_rnns))]
        current_context_vec = inputs.data.new(B, 256).zero_()
        stopnet_rnn_hidden = inputs.data.new(B, self.r * self.memory_dim).zero_()
        # Time first (T_decoder, B, memory_dim)
        if memory is not None:
            memory = memory.transpose(0, 1)
        outputs = []
        alignments = []
        stop_tokens = []
        t = 0
        memory_input = initial_memory
        while True:
            if t > 0:
                if greedy:
                    memory_input = outputs[-1]
                else:
                    memory_input = memory[t-1]
            # Prenet
            processed_memory = self.prenet(memory_input)
            # Attention RNN
            attention_rnn_hidden, current_context_vec, alignment = self.attention_rnn(
                processed_memory, current_context_vec, attention_rnn_hidden, inputs)
            # Concat RNN output and attention context vector
            decoder_input = self.project_to_decoder_in(
                torch.cat((attention_rnn_hidden, current_context_vec), -1))
            # Pass through the decoder RNNs
            for idx in range(len(self.decoder_rnns)):
                decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                    decoder_input, decoder_rnn_hiddens[idx])
                # Residual connectinon
                decoder_input = decoder_rnn_hiddens[idx] + decoder_input
            decoder_output = decoder_input
            # predict mel vectors from decoder vectors
            output = self.proj_to_mel(decoder_output)
            stop_input = output
            # predict stop token
            stop_token, stopnet_rnn_hidden = self.stopnet(stop_input, stopnet_rnn_hidden)
            outputs += [output]
            alignments += [alignment]
            stop_tokens += [stop_token]
            t += 1
            if (not greedy and self.training) or (greedy and memory is not None):
                if t >= T_decoder:
                    break
            else:
                if t > inputs.shape[1]/2 and stop_token > 0.8:
                    break
                elif t > self.max_decoder_steps:
                    print(" !! Decoder stopped with 'max_decoder_steps'. \
                          Something is probably wrong.")
                    break
        assert greedy or len(outputs) == T_decoder
        # Back to batch first
        alignments = torch.stack(alignments).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        return outputs, alignments, stop_tokens

    
class StopNet(nn.Module):
    r"""
    Predicting stop-token in decoder.
    
    Args:
        r (int): number of output frames of the network.
        memory_dim (int): feature dimension for each output frame.
    """
    
    def __init__(self, r, memory_dim):
        super(StopNet, self).__init__()
        self.rnn = nn.GRUCell(memory_dim * r, memory_dim * r)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(r * memory_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, inputs, rnn_hidden):
        """
        Args:
            inputs: network output tensor with r x memory_dim feature dimension.
            rnn_hidden: hidden state of the RNN cell.
        """
        rnn_hidden = self.rnn(inputs, rnn_hidden)
        outputs = self.relu(rnn_hidden)
        outputs = self.linear(outputs)
        outputs = self.sigmoid(outputs)
        return outputs, rnn_hidden
        