from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F
from .common_layers import Attention, Prenet, Linear, LinearBN


class ConvBNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, nonlinear=None):
        super(ConvBNBlock, self).__init__()
        assert (kernel_size - 1) % 2 == 0
        padding = (kernel_size - 1) // 2
        conv1d = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding)
        norm = nn.BatchNorm1d(out_channels)
        dropout = nn.Dropout(p=0.5)
        if nonlinear == 'relu':
            self.net = nn.Sequential(conv1d, norm, nn.ReLU(), dropout)
        elif nonlinear == 'tanh':
            self.net = nn.Sequential(conv1d, norm, nn.Tanh(), dropout)
        else:
            self.net = nn.Sequential(conv1d, norm, dropout)

    def forward(self, x):
        output = self.net(x)
        return output


class Postnet(nn.Module):
    def __init__(self, mel_dim, num_convs=5):
        super(Postnet, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            ConvBNBlock(mel_dim, 512, kernel_size=5, nonlinear='tanh'))
        for i in range(1, num_convs - 1):
            self.convolutions.append(
                ConvBNBlock(512, 512, kernel_size=5, nonlinear='tanh'))
        self.convolutions.append(
            ConvBNBlock(512, mel_dim, kernel_size=5, nonlinear=None))

    def forward(self, x):
        for layer in self.convolutions:
            x = layer(x)
        return x


class Encoder(nn.Module):
    def __init__(self, in_features=512):
        super(Encoder, self).__init__()
        convolutions = []
        for _ in range(3):
            convolutions.append(
                ConvBNBlock(in_features, in_features, 5, 'relu'))
        self.convolutions = nn.Sequential(*convolutions)
        self.lstm = nn.LSTM(
            in_features,
            int(in_features / 2),
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.rnn_state = None

    def forward(self, x, input_lengths):
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        input_lengths = input_lengths.cpu().numpy()
        x = nn.utils.rnn.pack_padded_sequence(
            x, input_lengths, batch_first=True)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            outputs,
            batch_first=True,
        )
        return outputs

    def inference(self, x):
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, _ = self.lstm(x)
        return outputs

    def inference_truncated(self, x):
        """
        Preserve encoder state for continuous inference
        """
        x = self.convolutions(x)
        x = x.transpose(1, 2)
        self.lstm.flatten_parameters()
        outputs, self.rnn_state = self.lstm(x, self.rnn_state)
        return outputs


# adapted from https://github.com/NVIDIA/tacotron2/
class Decoder(nn.Module):
    def __init__(self, in_features, inputs_dim, r, attn_win, attn_norm,
                 prenet_type, prenet_dropout, forward_attn, trans_agent,
                 location_attn, separate_stopnet):
        super(Decoder, self).__init__()
        self.mel_channels = inputs_dim
        self.r = r
        self.encoder_embedding_dim = in_features
        self.separate_stopnet = separate_stopnet
        self.attention_rnn_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(self.mel_channels * r, prenet_type,
                             prenet_dropout,
                             [self.prenet_dim, self.prenet_dim], bias=False)

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + in_features,
                                         self.attention_rnn_dim)

        self.attention_layer = Attention(self.attention_rnn_dim, in_features,
                                         128, location_attn, 32, 31, attn_win,
                                         attn_norm, forward_attn, trans_agent)

        self.decoder_rnn = nn.LSTMCell(self.attention_rnn_dim + in_features,
                                       self.decoder_rnn_dim, 1)

        self.linear_projection = Linear(self.decoder_rnn_dim + in_features,
                                        self.mel_channels * r)

        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            Linear(
                self.decoder_rnn_dim + self.mel_channels * r,
                1,
                bias=True,
                init_gain='sigmoid'))

        self.attention_rnn_init = nn.Embedding(1, self.attention_rnn_dim)
        self.go_frame_init = nn.Embedding(1, self.mel_channels * r)
        self.decoder_rnn_inits = nn.Embedding(1, self.decoder_rnn_dim)
        self.memory_truncated = None

    def get_go_frame(self, inputs):
        B = inputs.size(0)
        memory = self.go_frame_init(inputs.data.new_zeros(B).long())
        return memory

    def _init_states(self, inputs, mask, keep_states=False):
        B = inputs.size(0)
        T = inputs.size(1)

        if not keep_states:
            self.attention_hidden = self.attention_rnn_init(
                inputs.data.new_zeros(B).long())
            self.attention_cell = Variable(
                inputs.data.new(B, self.attention_rnn_dim).zero_())

            self.decoder_hidden = self.decoder_rnn_inits(
                inputs.data.new_zeros(B).long())
            self.decoder_cell = Variable(
                inputs.data.new(B, self.decoder_rnn_dim).zero_())

            self.context = Variable(
                inputs.data.new(B, self.encoder_embedding_dim).zero_())

        self.inputs = inputs
        self.processed_inputs = self.attention_layer.inputs_layer(inputs)
        self.mask = mask

    def _reshape_memory(self, memories):
        memories = memories.view(
            memories.size(0), int(memories.size(1) / self.r), -1)
        memories = memories.transpose(0, 1)
        return memories

    def _parse_outputs(self, outputs, stop_tokens, alignments):
        alignments = torch.stack(alignments).transpose(0, 1)
        stop_tokens = torch.stack(stop_tokens).transpose(0, 1)
        stop_tokens = stop_tokens.contiguous()
        outputs = torch.stack(outputs).transpose(0, 1).contiguous()
        outputs = outputs.view(outputs.size(0), -1, self.mel_channels)
        outputs = outputs.transpose(1, 2)
        return outputs, stop_tokens, alignments

    def decode(self, memory):
        cell_input = torch.cat((memory, self.context), -1)
        self.attention_hidden, self.attention_cell = self.attention_rnn(
            cell_input, (self.attention_hidden, self.attention_cell))
        self.attention_hidden = F.dropout(
            self.attention_hidden, self.p_attention_dropout, self.training)
        self.attention_cell = F.dropout(
            self.attention_cell, self.p_attention_dropout, self.training)

        self.context = self.attention_layer(self.attention_hidden, self.inputs,
                                            self.processed_inputs, self.mask)

        memory = torch.cat((self.attention_hidden, self.context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            memory, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell,
                                      self.p_decoder_dropout, self.training)

        decoder_hidden_context = torch.cat((self.decoder_hidden, self.context),
                                           dim=1)

        decoder_output = self.linear_projection(decoder_hidden_context)

        stopnet_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)

        if self.separate_stopnet:
            stop_token = self.stopnet(stopnet_input.detach())
        else:
            stop_token = self.stopnet(stopnet_input)
        return decoder_output, stop_token, self.attention_layer.attention_weights

    def forward(self, inputs, memories, mask):
        memory = self.get_go_frame(inputs).unsqueeze(0)
        memories = self._reshape_memory(memories)
        memories = torch.cat((memory, memories), dim=0)
        memories = self.prenet(memories)

        self._init_states(inputs, mask=mask)
        self.attention_layer.init_states(inputs)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < memories.size(0) - 1:
            memory = memories[len(outputs)]
            mel_output, stop_token, attention_weights = self.decode(memory)
            outputs += [mel_output.squeeze(1)]
            stop_tokens += [stop_token.squeeze(1)]
            alignments += [attention_weights]

        outputs, stop_tokens, alignments = self._parse_outputs(
            outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments

    def inference(self, inputs):
        memory = self.get_go_frame(inputs)
        self._init_states(inputs, mask=None)

        self.attention_layer.init_win_idx()
        self.attention_layer.init_states(inputs)

        outputs, stop_tokens, alignments, t = [], [], [], 0
        stop_flags = [True, False, False]
        stop_count = 0
        while True:
            memory = self.prenet(memory)
            mel_output, stop_token, alignment = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [mel_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            stop_flags[0] = stop_flags[0] or stop_token > 0.5
            stop_flags[1] = stop_flags[1] or (alignment[0, -2:].sum() > 0.8
                                              and t > inputs.shape[1])
            stop_flags[2] = t > inputs.shape[1] * 2
            if all(stop_flags):
                stop_count += 1
                if stop_count > 20:
                    break
            elif len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

            memory = mel_output
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(
            outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments

    def inference_truncated(self, inputs):
        """
        Preserve decoder states for continuous inference
        """
        if self.memory_truncated is None:
            self.memory_truncated = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None, keep_states=False)
        else:
            self._init_states(inputs, mask=None, keep_states=True)

        self.attention_layer.init_win_idx()
        self.attention_layer.init_states(inputs)
        outputs, stop_tokens, alignments, t = [], [], [], 0
        stop_flags = [True, False, False]
        stop_count = 0
        while True:
            memory = self.prenet(self.memory_truncated)
            mel_output, stop_token, alignment = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [mel_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            stop_flags[0] = stop_flags[0] or stop_token > 0.5
            stop_flags[1] = stop_flags[1] or (alignment[0, -2:].sum() > 0.8
                                              and t > inputs.shape[1])
            stop_flags[2] = t > inputs.shape[1] * 2
            if all(stop_flags):
                stop_count += 1
                if stop_count > 20:
                    break
            elif len(outputs) == self.max_decoder_steps:
                print("   | > Decoder stopped with 'max_decoder_steps")
                break

            self.memory_truncated = mel_output
            t += 1

        outputs, stop_tokens, alignments = self._parse_outputs(
            outputs, stop_tokens, alignments)

        return outputs, stop_tokens, alignments

    def inference_step(self, inputs, t, memory=None):
        """
        For debug purposes
        """
        if t == 0:
            memory = self.get_go_frame(inputs)
            self._init_states(inputs, mask=None)

        memory = self.prenet(memory)
        mel_output, stop_token, alignment = self.decode(memory)
        stop_token = torch.sigmoid(stop_token.data)
        memory = mel_output
        return mel_output, stop_token, alignment
