import torch
from scipy.stats import betabinom
from torch import nn
from torch.nn import functional as F

from TTS.tts.layers.tacotron.common_layers import Linear


class LocationLayer(nn.Module):
    """Layers for Location Sensitive Attention

    Args:
        attention_dim (int): number of channels in the input tensor.
        attention_n_filters (int, optional): number of filters in convolution. Defaults to 32.
        attention_kernel_size (int, optional): kernel size of convolution filter. Defaults to 31.
    """

    def __init__(self, attention_dim, attention_n_filters=32, attention_kernel_size=31):
        super().__init__()
        self.location_conv1d = nn.Conv1d(
            in_channels=2,
            out_channels=attention_n_filters,
            kernel_size=attention_kernel_size,
            stride=1,
            padding=(attention_kernel_size - 1) // 2,
            bias=False,
        )
        self.location_dense = Linear(attention_n_filters, attention_dim, bias=False, init_gain="tanh")

    def forward(self, attention_cat):
        """
        Shapes:
            attention_cat: [B, 2, C]
        """
        processed_attention = self.location_conv1d(attention_cat)
        processed_attention = self.location_dense(processed_attention.transpose(1, 2))
        return processed_attention


class GravesAttention(nn.Module):
    """Graves Attention as is ref1 with updates from ref2.
    ref1: https://arxiv.org/abs/1910.10288
    ref2: https://arxiv.org/pdf/1906.01083.pdf

    Args:
        query_dim (int): number of channels in query tensor.
        K (int): number of Gaussian heads to be used for computing attention.
    """

    COEF = 0.3989422917366028  # numpy.sqrt(1/(2*numpy.pi))

    def __init__(self, query_dim, K):
        super().__init__()
        self._mask_value = 1e-8
        self.K = K
        # self.attention_alignment = 0.05
        self.eps = 1e-5
        self.J = None
        self.N_a = nn.Sequential(
            nn.Linear(query_dim, query_dim, bias=True), nn.ReLU(), nn.Linear(query_dim, 3 * K, bias=True)
        )
        self.attention_weights = None
        self.mu_prev = None
        self.init_layers()

    def init_layers(self):
        torch.nn.init.constant_(self.N_a[2].bias[(2 * self.K) : (3 * self.K)], 1.0)  # bias mean
        torch.nn.init.constant_(self.N_a[2].bias[self.K : (2 * self.K)], 10)  # bias std

    def init_states(self, inputs):
        if self.J is None or inputs.shape[1] + 1 > self.J.shape[-1]:
            self.J = torch.arange(0, inputs.shape[1] + 2.0).to(inputs.device) + 0.5
        self.attention_weights = torch.zeros(inputs.shape[0], inputs.shape[1]).to(inputs.device)
        self.mu_prev = torch.zeros(inputs.shape[0], self.K).to(inputs.device)

    # pylint: disable=R0201
    # pylint: disable=unused-argument
    def preprocess_inputs(self, inputs):
        return None

    def forward(self, query, inputs, processed_inputs, mask):
        """
        Shapes:
            query: [B, C_attention_rnn]
            inputs: [B, T_in, C_encoder]
            processed_inputs: place_holder
            mask: [B, T_in]
        """
        gbk_t = self.N_a(query)
        gbk_t = gbk_t.view(gbk_t.size(0), -1, self.K)

        # attention model parameters
        # each B x K
        g_t = gbk_t[:, 0, :]
        b_t = gbk_t[:, 1, :]
        k_t = gbk_t[:, 2, :]

        # dropout to decorrelate attention heads
        g_t = torch.nn.functional.dropout(g_t, p=0.5, training=self.training)

        # attention GMM parameters
        sig_t = torch.nn.functional.softplus(b_t) + self.eps

        mu_t = self.mu_prev + torch.nn.functional.softplus(k_t)
        g_t = torch.softmax(g_t, dim=-1) + self.eps

        j = self.J[: inputs.size(1) + 1]

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
    """Bahdanau Attention with various optional modifications.
    - Location sensitive attnetion: https://arxiv.org/abs/1712.05884
    - Forward Attention: https://arxiv.org/abs/1807.06736 + state masking at inference
    - Using sigmoid instead of softmax normalization
    - Attention windowing at inference time

    Note:
        Location Sensitive Attention extends the additive attention mechanism
    to use cumulative attention weights from previous decoder time steps with the current time step features.

        Forward attention computes most probable monotonic alignment. The modified attention probabilities at each
    timestep are computed recursively by the forward algorithm.

        Transition agent in the forward attention explicitly gates the attention mechanism whether to move forward or
    stay at each decoder timestep.

        Attention windowing is a inductive prior that prevents the model from attending to previous and future timesteps
    beyond a certain window.

    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the vakue tensor. In general, the value tensor is the output of the encoder layer.
        attention_dim (int): number of channels of the inner attention layers.
        location_attention (bool): enable/disable location sensitive attention.
        attention_location_n_filters (int): number of location attention filters.
        attention_location_kernel_size (int): filter size of location attention convolution layer.
        windowing (int): window size for attention windowing. if it is 5, for computing the attention, it only considers the time steps [(t-5), ..., (t+5)] of the input.
        norm (str): normalization method applied to the attention weights. 'softmax' or 'sigmoid'
        forward_attn (bool): enable/disable forward attention.
        trans_agent (bool): enable/disable transition agent in the forward attention.
        forward_attn_mask (int): enable/disable an explicit masking in forward attention. It is useful to set at especially inference time.
    """

    # Pylint gets confused by PyTorch conventions here
    # pylint: disable=attribute-defined-outside-init
    def __init__(
        self,
        query_dim,
        embedding_dim,
        attention_dim,
        location_attention,
        attention_location_n_filters,
        attention_location_kernel_size,
        windowing,
        norm,
        forward_attn,
        trans_agent,
        forward_attn_mask,
    ):
        super().__init__()
        self.query_layer = Linear(query_dim, attention_dim, bias=False, init_gain="tanh")
        self.inputs_layer = Linear(embedding_dim, attention_dim, bias=False, init_gain="tanh")
        self.v = Linear(attention_dim, 1, bias=True)
        if trans_agent:
            self.ta = nn.Linear(query_dim + embedding_dim, 1, bias=True)
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
        self.alpha = torch.cat([torch.ones([B, 1]), torch.zeros([B, T])[:, :-1] + 1e-7], dim=1).to(inputs.device)
        self.u = (0.5 * torch.ones([B, 1])).to(inputs.device)

    def init_location_attention(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights_cum = torch.zeros([B, T], device=inputs.device)

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
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
        attention_cat = torch.cat((self.attention_weights.unsqueeze(1), self.attention_weights_cum.unsqueeze(1)), dim=1)
        processed_query = self.query_layer(query.unsqueeze(1))
        processed_attention_weights = self.location_layer(attention_cat)
        energies = self.v(torch.tanh(processed_query + processed_attention_weights + processed_inputs))
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
        fwd_shifted_alpha = F.pad(self.alpha[:, :-1].clone().to(alignment.device), (1, 0, 0, 0))
        # compute transition potentials
        alpha = ((1 - self.u) * self.alpha + self.u * fwd_shifted_alpha + 1e-8) * alignment
        # force incremental alignment
        if not self.training and self.forward_attn_mask:
            _, n = fwd_shifted_alpha.max(1)
            val, _ = alpha.max(1)
            for b in range(alignment.shape[0]):
                alpha[b, n[b] + 3 :] = 0
                alpha[b, : (n[b] - 1)] = 0  # ignore all previous states to prevent repetition.
                alpha[b, (n[b] - 2)] = 0.01 * val[b]  # smoothing factor for the prev step
        # renormalize attention weights
        alpha = alpha / alpha.sum(dim=1, keepdim=True)
        return alpha

    def forward(self, query, inputs, processed_inputs, mask):
        """
        shapes:
            query: [B, C_attn_rnn]
            inputs: [B, T_en, D_en]
            processed_inputs: [B, T_en, D_attn]
            mask: [B, T_en]
        """
        if self.location_attention:
            attention, _ = self.get_location_attention(query, processed_inputs)
        else:
            attention, _ = self.get_attention(query, processed_inputs)
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
            alignment = torch.sigmoid(attention) / torch.sigmoid(attention).sum(dim=1, keepdim=True)
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


class MonotonicDynamicConvolutionAttention(nn.Module):
    """Dynamic convolution attention from
    https://arxiv.org/pdf/1910.10288.pdf


    query -> linear -> tanh -> linear ->|
                                        |                                            mask values
                                        v                                              |    |
               atten_w(t-1) -|-> conv1d_dynamic -> linear -|-> tanh -> + -> softmax -> * -> * -> context
                             |-> conv1d_static  -> linear -|           |
                             |-> conv1d_prior   -> log ----------------|

    query: attention rnn output.

    Note:
        Dynamic convolution attention is an alternation of the location senstive attention with
    dynamically computed convolution filters from the previous attention scores and a set of
    constraints to keep the attention alignment diagonal.
        DCA is sensitive to mixed precision training and might cause instable training.

    Args:
        query_dim (int): number of channels in the query tensor.
        embedding_dim (int): number of channels in the value tensor.
        static_filter_dim (int): number of channels in the convolution layer computing the static filters.
        static_kernel_size (int): kernel size for the convolution layer computing the static filters.
        dynamic_filter_dim (int): number of channels in the convolution layer computing the dynamic filters.
        dynamic_kernel_size (int): kernel size for the convolution layer computing the dynamic filters.
        prior_filter_len (int, optional): [description]. Defaults to 11 from the paper.
        alpha (float, optional): [description]. Defaults to 0.1 from the paper.
        beta (float, optional): [description]. Defaults to 0.9 from the paper.
    """

    def __init__(
        self,
        query_dim,
        embedding_dim,  # pylint: disable=unused-argument
        attention_dim,
        static_filter_dim,
        static_kernel_size,
        dynamic_filter_dim,
        dynamic_kernel_size,
        prior_filter_len=11,
        alpha=0.1,
        beta=0.9,
    ):
        super().__init__()
        self._mask_value = 1e-8
        self.dynamic_filter_dim = dynamic_filter_dim
        self.dynamic_kernel_size = dynamic_kernel_size
        self.prior_filter_len = prior_filter_len
        self.attention_weights = None
        # setup key and query layers
        self.query_layer = nn.Linear(query_dim, attention_dim)
        self.key_layer = nn.Linear(attention_dim, dynamic_filter_dim * dynamic_kernel_size, bias=False)
        self.static_filter_conv = nn.Conv1d(
            1,
            static_filter_dim,
            static_kernel_size,
            padding=(static_kernel_size - 1) // 2,
            bias=False,
        )
        self.static_filter_layer = nn.Linear(static_filter_dim, attention_dim, bias=False)
        self.dynamic_filter_layer = nn.Linear(dynamic_filter_dim, attention_dim)
        self.v = nn.Linear(attention_dim, 1, bias=False)

        prior = betabinom.pmf(range(prior_filter_len), prior_filter_len - 1, alpha, beta)
        self.register_buffer("prior", torch.FloatTensor(prior).flip(0))

    # pylint: disable=unused-argument
    def forward(self, query, inputs, processed_inputs, mask):
        """
        query: [B, C_attn_rnn]
        inputs: [B, T_en, D_en]
        processed_inputs: place holder.
        mask: [B, T_en]
        """
        # compute prior filters
        prior_filter = F.conv1d(
            F.pad(self.attention_weights.unsqueeze(1), (self.prior_filter_len - 1, 0)), self.prior.view(1, 1, -1)
        )
        prior_filter = torch.log(prior_filter.clamp_min_(1e-6)).squeeze(1)
        G = self.key_layer(torch.tanh(self.query_layer(query)))
        # compute dynamic filters
        dynamic_filter = F.conv1d(
            self.attention_weights.unsqueeze(0),
            G.view(-1, 1, self.dynamic_kernel_size),
            padding=(self.dynamic_kernel_size - 1) // 2,
            groups=query.size(0),
        )
        dynamic_filter = dynamic_filter.view(query.size(0), self.dynamic_filter_dim, -1).transpose(1, 2)
        # compute static filters
        static_filter = self.static_filter_conv(self.attention_weights.unsqueeze(1)).transpose(1, 2)
        alignment = (
            self.v(
                torch.tanh(self.static_filter_layer(static_filter) + self.dynamic_filter_layer(dynamic_filter))
            ).squeeze(-1)
            + prior_filter
        )
        # compute attention weights
        attention_weights = F.softmax(alignment, dim=-1)
        # apply masking
        if mask is not None:
            attention_weights.data.masked_fill_(~mask, self._mask_value)
        self.attention_weights = attention_weights
        # compute context
        context = torch.bmm(attention_weights.unsqueeze(1), inputs).squeeze(1)
        return context

    def preprocess_inputs(self, inputs):  # pylint: disable=no-self-use
        return None

    def init_states(self, inputs):
        B = inputs.size(0)
        T = inputs.size(1)
        self.attention_weights = torch.zeros([B, T], device=inputs.device)
        self.attention_weights[:, 0] = 1.0


def init_attn(
    attn_type,
    query_dim,
    embedding_dim,
    attention_dim,
    location_attention,
    attention_location_n_filters,
    attention_location_kernel_size,
    windowing,
    norm,
    forward_attn,
    trans_agent,
    forward_attn_mask,
    attn_K,
):
    if attn_type == "original":
        return OriginalAttention(
            query_dim,
            embedding_dim,
            attention_dim,
            location_attention,
            attention_location_n_filters,
            attention_location_kernel_size,
            windowing,
            norm,
            forward_attn,
            trans_agent,
            forward_attn_mask,
        )
    if attn_type == "graves":
        return GravesAttention(query_dim, attn_K)
    if attn_type == "dynamic_convolution":
        return MonotonicDynamicConvolutionAttention(
            query_dim,
            embedding_dim,
            attention_dim,
            static_filter_dim=8,
            static_kernel_size=21,
            dynamic_filter_dim=8,
            dynamic_kernel_size=21,
            prior_filter_len=11,
            alpha=0.1,
            beta=0.9,
        )

    raise RuntimeError(f" [!] Given Attention Type '{attn_type}' is not exist.")
