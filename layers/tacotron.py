# coding: utf-8
import torch
from torch import nn
from .attention import AttentionRNNCell


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
        self.layers = nn.ModuleList([
            nn.Linear(in_size, out_size)
            for (in_size, out_size) in zip(in_features, out_features)
        ])
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        # self.init_layers()

    def init_layers(self):
        for layer in self.layers:
            torch.nn.init.xavier_uniform_(
                layer.weight, gain=torch.nn.init.calculate_gain('relu'))

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

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride,
                 padding,
                 activation=None):

        super(BatchNormConv1d, self).__init__()
        self.padding = padding
        self.padder = nn.ConstantPad1d(padding, 0)
        self.conv1d = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=0,
            bias=False)
        # Following tensorflow's default parameters
        self.bn = nn.BatchNorm1d(out_channels, momentum=0.99, eps=1e-3)
        self.activation = activation
        # self.init_layers()

    def init_layers(self):
        if type(self.activation) == torch.nn.ReLU:
            w_gain = 'relu'
        elif type(self.activation) == torch.nn.Tanh:
            w_gain = 'tanh'
        elif self.activation is None:
            w_gain = 'linear'
        else:
            raise RuntimeError('Unknown activation function')
        torch.nn.init.xavier_uniform_(
            self.conv1d.weight, gain=torch.nn.init.calculate_gain(w_gain))

    def forward(self, x):
        x = self.padder(x)
        x = self.conv1d(x)
        x = self.bn(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Highway(nn.Module):
    def __init__(self, in_size, out_size):
        super(Highway, self).__init__()
        self.H = nn.Linear(in_size, out_size)
        self.H.bias.data.zero_()
        self.T = nn.Linear(in_size, out_size)
        self.T.bias.data.fill_(-1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.H.weight, gain=torch.nn.init.calculate_gain('relu'))
        torch.nn.init.xavier_uniform_(
            self.T.weight, gain=torch.nn.init.calculate_gain('sigmoid'))

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

    def __init__(self,
                 in_features,
                 K=16,
                 conv_bank_features=128,
                 conv_projections=[128, 128],
                 highway_features=128,
                 gru_features=128,
                 num_highways=4):
        super(CBHG, self).__init__()
        self.in_features = in_features
        self.conv_bank_features = conv_bank_features
        self.highway_features = highway_features
        self.gru_features = gru_features
        self.conv_projections = conv_projections
        self.relu = nn.ReLU()
        # list of conv1d bank with filter size k=1...K
        # TODO: try dilational layers instead
        self.conv1d_banks = nn.ModuleList([
            BatchNormConv1d(
                in_features,
                conv_bank_features,
                kernel_size=k,
                stride=1,
                padding=[(k - 1) // 2, k // 2],
                activation=self.relu) for k in range(1, K + 1)
        ])
        # max pooling of conv bank, with padding
        # TODO: try average pooling OR larger kernel size
        self.max_pool1d = nn.Sequential(
            nn.ConstantPad1d([0, 1], value=0),
            nn.MaxPool1d(kernel_size=2, stride=1, padding=0))
        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]
        # setup conv1d projection layers
        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections,
                                           activations):
            layer = BatchNormConv1d(
                in_size,
                out_size,
                kernel_size=3,
                stride=1,
                padding=[1, 1],
                activation=ac)
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)
        # setup Highway layers
        if self.highway_features != conv_projections[-1]:
            self.pre_highway = nn.Linear(
                conv_projections[-1], highway_features, bias=False)
        self.highways = nn.ModuleList([
            Highway(highway_features, highway_features)
            for _ in range(num_highways)
        ])
        # bi-directional GPU layer
        self.gru = nn.GRU(
            gru_features,
            gru_features,
            1,
            batch_first=True,
            bidirectional=True)

    def forward(self, inputs):
        # (B, T_in, in_features)
        x = inputs
        # Needed to perform conv1d on time-axis
        # (B, in_features, T_in)
        if x.size(-1) == self.in_features:
            x = x.transpose(1, 2)
        T = x.size(-1)
        # (B, hid_features*K, T_in)
        # Concat conv1d bank outputs
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)
        x = self.max_pool1d(x)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        # (B, T_in, hid_feature)
        x = x.transpose(1, 2)
        # Back to the original shape
        x += inputs
        if self.highway_features != self.conv_projections[-1]:
            x = self.pre_highway(x)
        # Residual connection
        # TODO: try residual scaling as in Deep Voice 3
        # TODO: try plain residual layers
        for highway in self.highways:
            x = highway(x)
        # (B, T_in, hid_features*2)
        # TODO: replace GRU with convolution as in Deep Voice 3
        self.gru.flatten_parameters()
        outputs, _ = self.gru(x)
        return outputs


class EncoderCBHG(nn.Module):
    def __init__(self):
        super(EncoderCBHG, self).__init__()
        self.cbhg = CBHG(
            128,
            K=16,
            conv_bank_features=128,
            conv_projections=[128, 128],
            highway_features=128,
            gru_features=128,
            num_highways=4)

    def forward(self, x):
        return self.cbhg(x)


class Encoder(nn.Module):
    r"""Encapsulate Prenet and CBHG modules for encoder"""

    def __init__(self, in_features):
        super(Encoder, self).__init__()
        self.prenet = Prenet(in_features, out_features=[256, 128])
        self.cbhg = EncoderCBHG()

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


class PostCBHG(nn.Module):
    def __init__(self, mel_dim):
        super(PostCBHG, self).__init__()
        self.cbhg = CBHG(
            mel_dim,
            K=8,
            conv_bank_features=128,
            conv_projections=[256, mel_dim],
            highway_features=128,
            gru_features=128,
            num_highways=4)

    def forward(self, x):
        return self.cbhg(x)


class Decoder(nn.Module):
    r"""Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
        memory_size (int): size of the past window. if <= 0 memory_size = r
    """

    def __init__(self, in_features, memory_dim, r, memory_size,
                 attn_windowing, attn_norm):
        super(Decoder, self).__init__()
        self.r = r
        self.in_features = in_features
        self.max_decoder_steps = 500
        self.memory_size = memory_size if memory_size > 0 else r
        self.memory_dim = memory_dim
        # memory -> |Prenet| -> processed_memory
        self.prenet = Prenet(
            memory_dim * self.memory_size, out_features=[256, 128])
        # processed_inputs, processed_memory -> |Attention| -> Attention, attention, RNN_State
        self.attention_rnn = AttentionRNNCell(
            out_dim=128,
            rnn_dim=256,
            annot_dim=in_features,
            memory_dim=128,
            align_model='ls',
            windowing=attn_windowing,
            norm=attn_norm)
        # (processed_memory | attention context) -> |Linear| -> decoder_RNN_input
        self.project_to_decoder_in = nn.Linear(256 + in_features, 256)
        # decoder_RNN_input -> |RNN| -> RNN_state
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(256, memory_dim * r)
        # learn init values instead of zero init.
        self.attention_rnn_init = nn.Embedding(1, 256)
        self.memory_init = nn.Embedding(1, self.memory_size * memory_dim)
        self.decoder_rnn_inits = nn.Embedding(2, 256)
        self.stopnet = StopNet(256 + memory_dim * r)
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.project_to_decoder_in.weight,
            gain=torch.nn.init.calculate_gain('linear'))
        torch.nn.init.xavier_uniform_(
            self.proj_to_mel.weight,
            gain=torch.nn.init.calculate_gain('linear'))

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        B = memory.shape[0]
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.memory_dim:
            memory = memory.contiguous()
            memory = memory.view(B, memory.size(1) // self.r, -1)
        # Time first (T_decoder, B, memory_dim)
        memory = memory.transpose(0, 1)
        return memory

    def _init_states(self, inputs):
        """
        Initialization of decoder states
        """
        B = inputs.size(0)
        T = inputs.size(1)
        # go frame as zeros matrix
        self.memory_input = self.memory_init(inputs.data.new_zeros(B).long())

        # decoder states
        self.attention_rnn_hidden = self.attention_rnn_init(
            inputs.data.new_zeros(B).long())
        self.decoder_rnn_hiddens = [
            self.decoder_rnn_inits(inputs.data.new_tensor([idx] * B).long())
            for idx in range(len(self.decoder_rnns))
        ]
        self.current_context_vec = inputs.data.new(B, self.in_features).zero_()
        # attention states
        self.attention = inputs.data.new(B, T).zero_()
        self.attention_cum = inputs.data.new(B, T).zero_()

    def _parse_outputs(self, outputs, attentions, stop_tokens):
        # Back to batch first
        attentions = torch.stack(attentions).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1).squeeze(-1)
        return outputs, attentions, stop_tokens

    def decode(self,
               inputs,
               t, 
               mask=None):
        # Prenet
        processed_memory = self.prenet(self.memory_input)
        # Attention RNN
        attention_cat = torch.cat(
            (self.attention.unsqueeze(1), self.attention_cum.unsqueeze(1)), dim=1)
        self.attention_rnn_hidden, self.current_context_vec, self.attention = self.attention_rnn(
            processed_memory, self.current_context_vec, self.attention_rnn_hidden,
            inputs, attention_cat, mask, t)
        del attention_cat
        self.attention_cum += self.attention
        # Concat RNN output and attention context vector
        decoder_input = self.project_to_decoder_in(
            torch.cat((self.attention_rnn_hidden, self.current_context_vec), -1))
        # Pass through the decoder RNNs
        for idx in range(len(self.decoder_rnns)):
            self.decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                decoder_input, self.decoder_rnn_hiddens[idx])
            # Residual connection
            decoder_input = self.decoder_rnn_hiddens[idx] + decoder_input
        decoder_output = decoder_input
        del decoder_input
        # predict mel vectors from decoder vectors
        output = self.proj_to_mel(decoder_output)
        output = torch.sigmoid(output)
        # predict stop token
        stopnet_input = torch.cat([decoder_output, output], -1)
        del decoder_output
        stop_token = self.stopnet(stopnet_input)
        return output, stop_token, self.attention

    def _update_memory_queue(self, new_memory):
        if self.memory_size > 0:
            self.memory_input = torch.cat([
                self.memory_input[:, self.r * self.memory_dim:].clone(),
                new_memory
            ], dim=-1)
        else:
            self.memory_input = new_memory

    def forward(self, inputs, memory, mask):
        """
        Args:
            inputs: Encoder outputs.
            memory: Decoder memory (autoregression. If None (at eval-time),
              decoder outputs are used as decoder inputs. If None, it uses the last
              output as the input.
            mask: Attention mask for sequence padding.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - memory: batch x #mel_specs x mel_spec_dim
        """
        # Run greedy decoding if memory is None
        memory = self._reshape_memory(memory)
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        while len(outputs) < memory.size(0):
            if t > 0:
                new_memory = memory[t - 1]
                self._update_memory_queue(new_memory)
            output, stop_token, attention = self.decode(inputs, t, mask)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1

        return self._parse_outputs(outputs, attentions, stop_tokens)

    def inference(self, inputs):
        """
        Args:
            inputs: Encoder outputs.

        Shapes:
            - inputs: batch x time x encoder_out_dim
        """
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        while True:
            if t > 0:
                new_memory = outputs[-1]
                self._update_memory_queue(new_memory)
            output, stop_token, attention = self.decode(inputs, t, None)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token]
            t += 1
            if t > inputs.shape[1] / 4 and (stop_token > 0.6
                                            or attention[:, -1].item() > 0.6):
                break
            elif t > self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break
        return self._parse_outputs(outputs, attentions, stop_tokens)


class StopNet(nn.Module):
    r"""
    Args:
        in_features (int): feature dimension of input.
    """

    def __init__(self, in_features):
        super(StopNet, self).__init__()
        self.dropout = nn.Dropout(0.1)
        self.linear = nn.Linear(in_features, 1)
        torch.nn.init.xavier_uniform_(
            self.linear.weight, gain=torch.nn.init.calculate_gain('linear'))

    def forward(self, inputs):
        outputs = self.dropout(inputs)
        outputs = self.linear(outputs)
        return outputs
