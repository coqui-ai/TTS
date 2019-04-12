from math import sqrt
import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(
            in_features, out_features, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LinearBN(nn.Module):
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_gain='linear'):
        super(LinearBN, self).__init__()
        self.linear_layer = torch.nn.Linear(
            in_features, out_features, bias=bias)
        self.bn = nn.BatchNorm1d(out_features)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        out = self.linear_layer(x)
        if len(out.shape)==3:
            out = out.permute(1, 2, 0)
        out = self.bn(out)
        if len(out.shape) == 3:
            out = out.permute(2, 0, 1)
        return out


class Prenet(nn.Module):
    def __init__(self, in_features, prenet_type, out_features=[256, 256]):
        super(Prenet, self).__init__()
        self.prenet_type = prenet_type
        in_features = [in_features] + out_features[:-1]
        if prenet_type == "bn":
            self.layers = nn.ModuleList([
                LinearBN(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_features, out_features)
            ])
        elif prenet_type == "original":
            self.layers = nn.ModuleList(
                [Linear(in_size, out_size, bias=False)
                for (in_size, out_size) in zip(in_features, out_features)
            ])

    def forward(self, x):
        for linear in self.layers:
            if self.prenet_type == "original":
                x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
            elif self.prenet_type == "bn":
                x = F.relu(linear(x))
        return x
        

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


class LocationLayer(nn.Module):
    def __init__(self, attention_n_filters, attention_kernel_size,
                 attention_dim):
        super(LocationLayer, self).__init__()
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=31,
            stride=1,
            padding=(31 - 1) // 2,
            bias=False)
        self.location_dense = Linear(
            attention_n_filters, attention_dim, bias=False, init_gain='tanh')

    def forward(self, attention_cat):
        processed_attention = self.location_conv(attention_cat)
        processed_attention = self.location_dense(
            processed_attention.transpose(1, 2))
        return processed_attention


class Attention(nn.Module):
    def __init__(self, attention_rnn_dim, embedding_dim, attention_dim,
                 attention_location_n_filters, attention_location_kernel_size,
                 windowing, norm, forward_attn, trans_agent):
        super(Attention, self).__init__()
        self.query_layer = Linear(
            attention_rnn_dim, attention_dim, bias=False, init_gain='tanh')
        self.inputs_layer = Linear(
            embedding_dim, attention_dim, bias=False, init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(attention_dim + embedding_dim, 1, bias=True)
        self.location_layer = LocationLayer(attention_location_n_filters,
                                            attention_location_kernel_size,
                                            attention_dim)
        self._mask_value = -float("inf")
        self.windowing = windowing
        self.win_idx = None
        self.norm = norm
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent

    def init_win_idx(self):
        self.win_idx = -1
        self.win_back = 2
        self.win_front = 6
    
    def init_forward_attn_state(self, inputs):
        """
        Init forward attention states
        """
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.alpha = torch.cat([torch.ones([B, 1]), torch.zeros([B, T])[:, :-1]], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def get_attention(self, query, processed_inputs, attention_cat):
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights +
                       processed_inputs))

        energies = energies.squeeze(-1)
        return energies, processed_query

    def apply_windowing(self, attention, inputs):
        back_win = self.win_idx - self.win_back
        front_win = self.win_idx + self.win_front
        if back_win > 0:
            attention[:, :back_win] = -float("inf")
        if front_win < inputs.shape[1]:
            attention[:, front_win:] = -float("inf")
        # this is a trick to solve a special problem.
        # but it does not hurt.
        if self.win_idx == -1:
            attention[:, 0] = attention.max()
        # Update the window
        self.win_idx = torch.argmax(attention, 1).long()[0].item()
        return attention

    def apply_forward_attention(self, inputs, alignment, processed_query):
        # forward attention
        prev_alpha = F.pad(self.alpha[:, :-1].clone(), (1, 0, 0, 0)).to(inputs.device)
        self.alpha = (((1-self.u) * self.alpha.clone().to(inputs.device) + self.u * prev_alpha) + 1e-7) * alignment
        alpha_norm = self.alpha / self.alpha.sum(dim=1).unsqueeze(1)
        # compute context
        context = torch.bmm(alpha_norm.unsqueeze(1), inputs)
        context = context.squeeze(1)
        # compute transition agent
        if self.trans_agent:
            ta_input = torch.cat([context, processed_query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context, alpha_norm, alignment

    def forward(self, attention_hidden_state, inputs, processed_inputs,
                attention_cat, mask):
        attention, processed_query = self.get_attention(
            attention_hidden_state, processed_inputs, attention_cat)

        # apply masking
        if mask is not None:
            attention.data.masked_fill_(1 - mask, self._mask_value)
        # apply windowing - only in eval mode
        if not self.training and self.windowing:
            attention = self.apply_windowing(attention, inputs)
        # normalize attention values
        if self.norm == "softmax":
            alignment = torch.softmax(attention, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(attention) / torch.sigmoid(
                    attention).sum(dim=1).unsqueeze(1)
        else:
            raise RuntimeError("Unknown value for attention norm type")
        # apply forward attention if enabled
        if self.forward_attn:
            return self.apply_forward_attention(inputs, alignment, processed_query)
        else:
            context = torch.bmm(alignment.unsqueeze(1), inputs)
            context = context.squeeze(1)
            return context, alignment, alignment


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
    def __init__(self, in_features, inputs_dim, r, attn_win, attn_norm, prenet_type, forward_attn, trans_agent):
        super(Decoder, self).__init__()
        self.mel_channels = inputs_dim
        self.r = r
        self.encoder_embedding_dim = in_features
        self.attention_rnn_dim = 1024
        self.decoder_rnn_dim = 1024
        self.prenet_dim = 256
        self.max_decoder_steps = 1000
        self.gate_threshold = 0.5
        self.p_attention_dropout = 0.1
        self.p_decoder_dropout = 0.1

        self.prenet = Prenet(self.mel_channels * r, prenet_type,
                             [self.prenet_dim, self.prenet_dim])

        self.attention_rnn = nn.LSTMCell(self.prenet_dim + in_features,
                                         self.attention_rnn_dim)

        self.attention_layer = Attention(self.attention_rnn_dim, in_features,
                                         128, 32, 31, attn_win, attn_norm, forward_attn, trans_agent)

        self.decoder_rnn = nn.LSTMCell(self.attention_rnn_dim + in_features,
                                       self.decoder_rnn_dim, 1)

        self.linear_projection = Linear(self.decoder_rnn_dim + in_features,
                                        self.mel_channels * r)

        self.stopnet = nn.Sequential(
            nn.Dropout(0.1),
            Linear(self.decoder_rnn_dim + self.mel_channels * r,
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

        self.attention_weights = Variable(inputs.data.new(B, T).zero_())
        self.attention_weights_cum = Variable(inputs.data.new(B, T).zero_())
        
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
        outputs = outputs.view(
            outputs.size(0), -1, self.mel_channels)
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

        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
                                   self.attention_weights_cum.unsqueeze(1)),
                                  dim=1)
        self.context, self.attention_weights, alignments = self.attention_layer(
            self.attention_hidden, self.inputs, self.processed_inputs,
            attention_cat, self.mask)

        self.attention_weights_cum += alignments
        memory = torch.cat(
            (self.attention_hidden, self.context), -1)
        self.decoder_hidden, self.decoder_cell = self.decoder_rnn(
            memory, (self.decoder_hidden, self.decoder_cell))
        self.decoder_hidden = F.dropout(self.decoder_hidden,
                                        self.p_decoder_dropout, self.training)
        self.decoder_cell = F.dropout(self.decoder_cell,
                                      self.p_decoder_dropout, self.training)

        decoder_hidden_context = torch.cat(
            (self.decoder_hidden, self.context), dim=1)

        decoder_output = self.linear_projection(
            decoder_hidden_context)

        stopnet_input = torch.cat((self.decoder_hidden, decoder_output), dim=1)

        gate_prediction = self.stopnet(stopnet_input)
        return decoder_output, gate_prediction, self.attention_weights

    def forward(self, inputs, memories, mask):
        memory = self.get_go_frame(inputs).unsqueeze(0)
        memories = self._reshape_memory(memories)
        memories = torch.cat((memory, memories), dim=0)
        memories = self.prenet(memories)

        self._init_states(inputs, mask=mask)
        if self.attention_layer.forward_attn:
            self.attention_layer.init_forward_attn_state(inputs)

        outputs, stop_tokens, alignments = [], [], []
        while len(outputs) < memories.size(0) - 1:
            memory = memories[len(outputs)]
            mel_output, stop_token, attention_weights = self.decode(
                memory)
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
        if self.attention_layer.forward_attn:
            self.attention_layer.init_forward_attn_state(inputs)

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
            stop_flags[1] = stop_flags[1] or (alignment[0, -2:].sum() > 0.5 and t > inputs.shape[1])
            stop_flags[2] = t > inputs.shape[1] * 2
            if all(stop_flags):
                stop_count += 1
                if stop_count > 2:
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
        if self.attention_layer.forward_attn:
            self.attention_layer.init_forward_attn_state(inputs)
        outputs, stop_tokens, alignments, t = [], [], [], 0
        stop_flags = [False, False, False]
        stop_count = 0
        while True:
            memory = self.prenet(self.memory_truncated)
            mel_output, stop_token, alignment = self.decode(memory)
            stop_token = torch.sigmoid(stop_token.data)
            outputs += [mel_output.squeeze(1)]
            stop_tokens += [stop_token]
            alignments += [alignment]

            stop_flags[0] = stop_flags[0] or stop_token > 0.5
            stop_flags[1] = stop_flags[1] or (alignment[0, -2:].sum() > 0.5 and t > inputs.shape[1])
            stop_flags[2] = t > inputs.shape[1] * 2
            if all(stop_flags):
                stop_count += 1
                if stop_count > 2:
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
