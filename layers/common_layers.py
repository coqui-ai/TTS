import torch
from torch import nn
from torch.autograd import Variable
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


####################
# ATTENTION MODULES
####################


class LocationLayer(nn.Module):
    def __init__(self,
                 attention_dim,
                 attention_n_filters=32,
                 attention_kernel_size=31):
        super(LocationLayer, self).__init__()
        self.location_conv = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False)
        self.location_dense = Linear(
            attention_n_filters, attention_dim, bias=False, init_gain='tanh')

    def forward(self, attention_cat):
        processed_attention = self.location_conv(attention_cat)
        processed_attention = self.location_dense(
            processed_attention.transpose(1, 2))
        return processed_attention


class GravesAttention(nn.Module):
    """ Discretized Graves attention:
        - https://arxiv.org/abs/1910.10288
        - https://arxiv.org/pdf/1906.01083.pdf
    """
    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):
        super(GravesAttention, self).__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True),
            nn.ReLU(),
            nn.Linear(query_dim, 3*K, bias=True))
        self.attention_weights = None
        self.mu_prev = None
        self.init_layers()

    def init_layers(self):
        torch.nn.init.constant_(self.N_a[2].bias[(2*self.K):(3*self.K)], 1.)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K:(2*self.K)], 10)  # bias std

    def init_states(self, inputs):
        if self.J is None or inputs.shape[1]+1 > self.J.shape[-1]:
            self.J = torch.arange(0, inputs.shape[1]+2).to(inputs.device) + 0.5
        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument
    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs, processed_inputs, mask):
        """
        shapes:
            query: B x D_attention_rnn
            inputs: B x T_in x D_encoder
            processed_inputs: place_holder
            mask: B x T_in
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # attention GMM parameters
        sig_t = torch.nn.functional.softplus(b_t) + self.eps

        mu_t = self.mu_prev + torch.nn.functional.softplus(k_t)
        g_t = torch.softmax(g_t, dim=-1) + self.eps

        j = self.J[:inputs.size(1)+1]

        # attention weights
        phi_t = g_t.unsqueeze(-1) * (1 / (1 + torch.sigmoid((mu_t.unsqueeze(-1) - j) / sig_t.unsqueeze(-1))))

        # discritize attention weights
        alpha_t = torch.sum(phi_t, 1)
        alpha_t = alpha_t[:, 1:] - alpha_t[:, :-1]
        alpha_t[alpha_t == 0] = 1e-8

        # apply masking
        if mask is not None:
            alpha_t.data.masked_fill_(~mask, self._mask_value)

        context = torch.bmm(alpha_t.unsqueeze(1), inputs).squeeze(1)
        self.attention_weights = alpha_t
        self.mu_prev = mu_t
        return context


class OriginalAttention(nn.Module):
    """Following the methods proposed here:
        - https://arxiv.org/abs/1712.05884
        - https://arxiv.org/abs/1807.06736 + state masking at inference
        - Using sigmoid instead of softmax normalization
        - Attention windowing at inference time
    """
    # Pylint gets confused by PyTorch conventions here
    #pylint: disable=attribute-defined-outside-init
    def __init__(self, query_dim, embedding_dim, attention_dim,
                 location_attention, attention_location_n_filters,
                 attention_location_kernel_size, windowing, norm, forward_attn,
                 trans_agent, forward_attn_mask):
        super(OriginalAttention, self).__init__()
        self.query_layer = Linear(
            query_dim, attention_dim, bias=False, init_gain='tanh')
        self.inputs_layer = Linear(
            embedding_dim, attention_dim, bias=False, init_gain='tanh')
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(
                query_dim + embedding_dim, 1, bias=True)
        if location_attention:
            self.location_layer = LocationLayer(
                attention_dim,
                attention_location_n_filters,
                attention_location_kernel_size,
            )
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

    def preprocess_inputs(self, inputs):
        return self.inputs_layer(inputs)

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

    def apply_forward_attention(self, alignment):
        # forward attention
        fwd_shifted_alpha = F.pad(self.alpha[:, :-1].clone().to(alignment.device),
                            (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha
                 + self.u * fwd_shifted_alpha
                 + 1e-8) * alignment
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, n2 = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3:] = 0
                alpha[b, :(
                    n[b] - 1
                )] = 0  # ignore all previous states to prevent repetition.
                alpha[b,
                      (n[b] - 2
                       )] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, query, inputs, processed_inputs, mask):
        """
        shapes:
            query: B x D_attn_rnn
            inputs: B x T_en x D_en
            processed_inputs:: B x T_en x D_attn
            mask: B x T_en
        """
        if self.location_attention:
            attention, _ = self.get_location_attention(
                query, processed_inputs)
        else:
            attention, _ = self.get_attention(
                query, processed_inputs)
        # apply masking
        if mask is not None:
            attention.data.masked_fill_(~mask, self._mask_value)
        # apply windowing - only in eval mode
        if not self.training and self.windowing:
            attention = self.apply_windowing(attention, inputs)

        # normalize attention values
        if self.norm == "softmax":
            alignment = torch.softmax(attention, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(attention) / torch.sigmoid(
                attention).sum(
                    dim=1, keepdim=True)
        else:
            raise ValueError("Unknown value for attention norm type")

        if self.location_attention:
            self.update_location_attention(alignment)

        # apply forward attention if enabled
        if self.forward_attn:
            alignment = self.apply_forward_attention(alignment)
            self.alpha = alignment

        context = torch.bmm(alignment.unsqueeze(1), inputs)
        context = context.squeeze(1)
        self.attention_weights = alignment

        # compute transition agent
        if self.forward_attn and self.trans_agent:
            ta_input = torch.cat([context, query.squeeze(1)], dim=-1)
            self.u = torch.sigmoid(self.ta(ta_input))
        return context


def init_attn(attn_type, query_dim, embedding_dim, attention_dim,
              location_attention, attention_location_n_filters,
              attention_location_kernel_size, windowing, norm, forward_attn,
              trans_agent, forward_attn_mask, attn_K):
    if attn_type == "original":
        return OriginalAttention(query_dim, embedding_dim, attention_dim,
                                 location_attention,
                                 attention_location_n_filters,
                                 attention_location_kernel_size, windowing,
                                 norm, forward_attn, trans_agent,
                                 forward_attn_mask)
    if attn_type == "graves":
        return GravesAttention(query_dim, attn_K)
    raise RuntimeError(
        " [!] Given Attention Type '{attn_type}' is not exist.")
