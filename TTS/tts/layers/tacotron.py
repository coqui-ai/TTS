# coding: utf-8
import torch
from torch import nn
from .common_layers import Prenet, init_attn, Linear


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
    # TODO: Try GLU layer
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
            - input: B x D x T_in
            - output: B x T_in x D*2
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
            BatchNormConv1d(in_features,
                            conv_bank_features,
                            kernel_size=k,
                            stride=1,
                            padding=[(k - 1) // 2, k // 2],
                            activation=self.relu) for k in range(1, K + 1)
        ])
        # max pooling of conv bank, with padding
        # TODO: try average pooling OR larger kernel size
        out_features = [K * conv_bank_features] + conv_projections[:-1]
        activations = [self.relu] * (len(conv_projections) - 1)
        activations += [None]
        # setup conv1d projection layers
        layer_set = []
        for (in_size, out_size, ac) in zip(out_features, conv_projections,
                                           activations):
            layer = BatchNormConv1d(in_size,
                                    out_size,
                                    kernel_size=3,
                                    stride=1,
                                    padding=[1, 1],
                                    activation=ac)
            layer_set.append(layer)
        self.conv1d_projections = nn.ModuleList(layer_set)
        # setup Highway layers
        if self.highway_features != conv_projections[-1]:
            self.pre_highway = nn.Linear(conv_projections[-1],
                                         highway_features,
                                         bias=False)
        self.highways = nn.ModuleList([
            Highway(highway_features, highway_features)
            for _ in range(num_highways)
        ])
        # bi-directional GPU layer
        self.gru = nn.GRU(gru_features,
                          gru_features,
                          1,
                          batch_first=True,
                          bidirectional=True)

    def forward(self, inputs):
        # (B, in_features, T_in)
        x = inputs
        # (B, hid_features*K, T_in)
        # Concat conv1d bank outputs
        outs = []
        for conv1d in self.conv1d_banks:
            out = conv1d(x)
            outs.append(out)
        x = torch.cat(outs, dim=1)
        assert x.size(1) == self.conv_bank_features * len(self.conv1d_banks)
        for conv1d in self.conv1d_projections:
            x = conv1d(x)
        x += inputs
        x = x.transpose(1, 2)
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
        # B x T x prenet_dim
        outputs = self.prenet(inputs)
        outputs = self.cbhg(outputs.transpose(1, 2))
        return outputs


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
    """Decoder module.

    Args:
        in_features (int): input vector (encoder output) sample size.
        memory_dim (int): memory vector (prev. time-step output) sample size.
        r (int): number of outputs per time step.
        memory_size (int): size of the past window. if <= 0 memory_size = r
        TODO: arguments
    """

    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init

    def __init__(self, in_features, memory_dim, r, memory_size, attn_type, attn_windowing,
                 attn_norm, prenet_type, prenet_dropout, forward_attn,
                 trans_agent, forward_attn_mask, location_attn, attn_K,
                 separate_stopnet, speaker_embedding_dim):
        super(Decoder, self).__init__()
        self.r_init = r
        self.r = r
        self.in_features = in_features
        self.max_decoder_steps = 500
        self.use_memory_queue = memory_size > 0
        self.memory_size = memory_size if memory_size > 0 else r
        self.memory_dim = memory_dim
        self.separate_stopnet = separate_stopnet
        self.query_dim = 256
        # memory -> |Prenet| -> processed_memory
        prenet_dim = memory_dim * self.memory_size + speaker_embedding_dim if self.use_memory_queue else memory_dim + speaker_embedding_dim
        self.prenet = Prenet(
            prenet_dim,
            prenet_type,
            prenet_dropout,
            out_features=[256, 128])
        # processed_inputs, processed_memory -> |Attention| -> Attention, attention, RNN_State
        # attention_rnn generates queries for the attention mechanism
        self.attention_rnn = nn.GRUCell(in_features + 128, self.query_dim)

        self.attention = init_attn(attn_type=attn_type,
                                   query_dim=self.query_dim,
                                   embedding_dim=in_features,
                                   attention_dim=128,
                                   location_attention=location_attn,
                                   attention_location_n_filters=32,
                                   attention_location_kernel_size=31,
                                   windowing=attn_windowing,
                                   norm=attn_norm,
                                   forward_attn=forward_attn,
                                   trans_agent=trans_agent,
                                   forward_attn_mask=forward_attn_mask,
                                   attn_K=attn_K)
        # (processed_memory | attention context) -> |Linear| -> decoder_RNN_input
        self.project_to_decoder_in = nn.Linear(256 + in_features, 256)
        # decoder_RNN_input -> |RNN| -> RNN_state
        self.decoder_rnns = nn.ModuleList(
            [nn.GRUCell(256, 256) for _ in range(2)])
        # RNN_state -> |Linear| -> mel_spec
        self.proj_to_mel = nn.Linear(256, memory_dim * self.r_init)
        # learn init values instead of zero init.
        self.stopnet = StopNet(256 + memory_dim * self.r_init)

    def set_r(self, new_r):
        self.r = new_r

    def _reshape_memory(self, memory):
        """
        Reshape the spectrograms for given 'r'
        """
        # Grouping multiple frames if necessary
        if memory.size(-1) == self.memory_dim:
            memory = memory.view(memory.shape[0], memory.size(1) // self.r, -1)
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
        if self.use_memory_queue:
            self.memory_input = torch.zeros(1, device=inputs.device).repeat(B, self.memory_dim * self.memory_size)
        else:
            self.memory_input = torch.zeros(1, device=inputs.device).repeat(B, self.memory_dim)
        # decoder states
        self.attention_rnn_hidden = torch.zeros(1, device=inputs.device).repeat(B, 256)
        self.decoder_rnn_hiddens = [
            torch.zeros(1, device=inputs.device).repeat(B, 256)
            for idx in range(len(self.decoder_rnns))
        ]
        self.context_vec = inputs.data.new(B, self.in_features).zero_()
        # cache attention inputs
        self.processed_inputs = self.attention.preprocess_inputs(inputs)

    def _parse_outputs(self, outputs, attentions, stop_tokens):
        # Back to batch first
        attentions = torch.stack(attentions).transpose(0, 1)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(
            outputs.size(0), -1, self.memory_dim)
        outputs = outputs.transpose(1, 2)
        return outputs, attentions, stop_tokens

    def decode(self, inputs, mask=None):
        # Prenet
        processed_memory = self.prenet(self.memory_input)
        # Attention RNN
        self.attention_rnn_hidden = self.attention_rnn(
            torch.cat((processed_memory, self.context_vec), -1),
            self.attention_rnn_hidden)
        self.context_vec = self.attention(
            self.attention_rnn_hidden, inputs, self.processed_inputs, mask)
        # Concat RNN output and attention context vector
        decoder_input = self.project_to_decoder_in(
            torch.cat((self.attention_rnn_hidden, self.context_vec), -1))

        # Pass through the decoder RNNs
        for idx in range(len(self.decoder_rnns)):
            self.decoder_rnn_hiddens[idx] = self.decoder_rnns[idx](
                decoder_input, self.decoder_rnn_hiddens[idx])
            # Residual connection
            decoder_input = self.decoder_rnn_hiddens[idx] + decoder_input
        decoder_output = decoder_input

        # predict mel vectors from decoder vectors
        output = self.proj_to_mel(decoder_output)
        # output = torch.sigmoid(output)
        # predict stop token
        stopnet_input = torch.cat([decoder_output, output], -1)
        if self.separate_stopnet:
            stop_token = self.stopnet(stopnet_input.detach())
        else:
            stop_token = self.stopnet(stopnet_input)
        output = output[:, : self.r * self.memory_dim]
        return output, stop_token, self.attention.attention_weights

    def _update_memory_input(self, new_memory):
        if self.use_memory_queue:
            if self.memory_size > self.r:
                # memory queue size is larger than number of frames per decoder iter
                self.memory_input = torch.cat([
                    new_memory, self.memory_input[:, :(
                        self.memory_size - self.r) * self.memory_dim].clone()
                ], dim=-1)
            else:
                # memory queue size smaller than number of frames per decoder iter
                self.memory_input = new_memory[:, :self.memory_size * self.memory_dim]
        else:
            # use only the last frame prediction
            # assert new_memory.shape[-1] == self.r * self.memory_dim
            self.memory_input = new_memory[:, self.memory_dim * (self.r - 1):]

    def forward(self, inputs, memory, mask, speaker_embeddings=None):
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
        self.attention.init_states(inputs)
        while len(outputs) < memory.size(0):
            if t > 0:
                new_memory = memory[t - 1]
                self._update_memory_input(new_memory)
            if speaker_embeddings is not None:
                self.memory_input = torch.cat([self.memory_input, speaker_embeddings], dim=-1)
            output, stop_token, attention = self.decode(inputs, mask)
            outputs += [output]
            attentions += [attention]
            stop_tokens += [stop_token.squeeze(1)]
            t += 1
        return self._parse_outputs(outputs, attentions, stop_tokens)

    def inference(self, inputs, speaker_embeddings=None):
        """
        Args:
            inputs: encoder outputs.
            speaker_embeddings: speaker vectors.

        Shapes:
            - inputs: batch x time x encoder_out_dim
            - speaker_embeddings: batch x embed_dim
        """
        outputs = []
        attentions = []
        stop_tokens = []
        t = 0
        self._init_states(inputs)
        self.attention.init_win_idx()
        self.attention.init_states(inputs)
        while True:
            if t > 0:
                new_memory = outputs[-1]
                self._update_memory_input(new_memory)
            if speaker_embeddings is not None:
                self.memory_input = torch.cat([self.memory_input, speaker_embeddings], dim=-1)
            output, stop_token, attention = self.decode(inputs, None)
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
