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
        if len(out.shape) == 3:
            out = out.permute(1, 2, 0)
        out = self.bn(out)
        if len(out.shape) == 3:
            out = out.permute(2, 0, 1)
        return out


class Prenet(nn.Module):
    def __init__(self,
                 in_features,
                 prenet_type="original",
                 prenet_dropout=True,
                 out_features=[256, 256],
                 bias=True):
        super(Prenet, self).__init__()
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        in_features = [in_features] + out_features[:-1]
        if prenet_type == "bn":
            self.layers = nn.ModuleList([
                LinearBN(in_size, out_size, bias=bias)
                for (in_size, out_size) in zip(in_features, out_features)
            ])
        elif prenet_type == "original":
            self.layers = nn.ModuleList([
                Linear(in_size, out_size, bias=bias)
                for (in_size, out_size) in zip(in_features, out_features)
            ])

    def forward(self, x):
        for linear in self.layers:
            if self.prenet_dropout:
                x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
            else:
                x = F.relu(linear(x))
        return x


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
                 location_attention, attention_location_n_filters,
                 attention_location_kernel_size, windowing, norm, forward_attn,
                 trans_agent, forward_attn_mask):
        super(Attention, self).__init__()
        self.query_layer = Linear(
            attention_rnn_dim, attention_dim, bias=False, init_gain='tanh')
        self.inputs_layer = Linear(
            embedding_dim, attention_dim, bias=False, init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(
                attention_rnn_dim + embedding_dim, 1, bias=True)
        if location_attention:
            self.location_layer = LocationLayer(
                attention_location_n_filters, attention_location_kernel_size,
                attention_dim)
        self._mask_value = -float("inf")
        self.windowing = windowing
        self.win_idx = None
        self.norm = norm
        self.forward_attn = forward_attn
        self.trans_agent = trans_agent
        self.forward_attn_mask = forward_attn_mask
        self.location_attention = location_attention

    def init_win_idx(self):
        self.win_idx = -1
        self.win_back = 2
        self.win_front = 6

    def init_forward_attn(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.alpha = torch.cat(
            [torch.ones([B, 1]),
             torch.zeros([B, T])[:, :-1] + 1e-7], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def init_location_attention(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights_cum = Variable(inputs.data.new(B, T).zero_())

    def init_states(self, inputs):
        B = inputs.shape[0]
        T = inputs.shape[1]
        self.attention_weights = Variable(inputs.data.new(B, T).zero_())
        if self.location_attention:
            self.init_location_attention(inputs)
        if self.forward_attn:
            self.init_forward_attn(inputs)
        if self.windowing:
            self.init_win_idx()

    def update_location_attention(self, alignments):
        self.attention_weights_cum += alignments

    def get_location_attention(self, query, processed_inputs):
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1),
                                   self.attention_weights_cum.unsqueeze(1)),
                                  dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(
            torch.tanh(processed_query + processed_attention_weights +
                       processed_inputs))
        energies = energies.squeeze(-1)
        return energies, processed_query

    def get_attention(self, query, processed_inputs):
        processed_query = self.query_layer(query.unsqueeze(1))
        energies = self.v(torch.tanh(processed_query + processed_inputs))
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

    def apply_forward_attention(self, inputs, alignment, query):
        # forward attention
        prev_alpha = F.pad(self.alpha[:, :-1].clone(),
                           (1, 0, 0, 0)).to(inputs.device)
        # compute transition potentials
        alpha = (((1 - self.u) * self.alpha.clone().to(inputs.device) +
                  self.u * prev_alpha) + 1e-8) * alignment
        # force incremental alignment - TODO: make configurable
        if not self.training and self.forward_attn_mask:
            _, n = prev_alpha.max(1)
            val, n2 = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3:] = 0
                alpha[b, :(n[b] - 1)] = 0  # ignore all previous states to prevent repetition.
                alpha[b, (n[b] - 2)] = 0.01 * val[b]  # smoothing factor for the prev step
        # compute attention weights
        self.alpha = alpha / alpha.sum(dim=1).unsqueeze(1)
        # compute context
        context = torch.bmm(self.alpha.unsqueeze(1), inputs)
        context = context.squeeze(1)
        # compute transition agent
        if self.trans_agent:
            ta_input = torch.cat([context, query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context, self.alpha

    def forward(self, attention_hidden_state, inputs, processed_inputs, mask):
        if self.location_attention:
            attention, processed_query = self.get_location_attention(
                attention_hidden_state, processed_inputs)
        else:
            attention, processed_query = self.get_attention(
                attention_hidden_state, processed_inputs)
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
        if self.location_attention:
            self.update_location_attention(alignment)
        # apply forward attention if enabled
        if self.forward_attn:
            context, self.attention_weights = self.apply_forward_attention(
                inputs, alignment, attention_hidden_state)
        else:
            context = torch.bmm(alignment.unsqueeze(1), inputs)
            context = context.squeeze(1)
            self.attention_weights = alignment
        return context