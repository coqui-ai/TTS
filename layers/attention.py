import torch
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, hidden_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, hidden_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, hidden_dim, bias=True)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - query: (batch, 1, dim) or (batch, dim)
            - annots: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        # (batch, max_time, 1)
        alignment = self.v(nn.functional.tanh(
            processed_query + processed_annots))
        # (batch, max_time)
        return alignment.squeeze(-1)


def get_mask_from_lengths(inputs, inputs_lengths):
    """Get mask tensor from list of length

    Args:
        inputs: Tensor in size (batch, max_time, dim)
        inputs_lengths: array like
    """
    mask = inputs.data.new(inputs.size(0), inputs.size(1)).byte().zero_()
    for idx, l in enumerate(inputs_lengths):
        mask[idx][:l] = 1
    return ~mask


class AttentionRNN(nn.Module):
    def __init__(self, out_dim, annot_dim, memory_dim,
                 score_mask_value=-float("inf")):
        super(AttentionRNN, self).__init__()
        self.rnn_cell = nn.GRUCell(out_dim + memory_dim, out_dim)
        self.alignment_model = BahdanauAttention(annot_dim, out_dim, out_dim)
        self.score_mask_value = score_mask_value

    def forward(self, memory, context, rnn_state, annotations,
                mask=None, annotations_lengths=None):
        if annotations_lengths is not None and mask is None:
            mask = get_mask_from_lengths(annotations, annotations_lengths)
        # Concat input query and previous context context
        rnn_input = torch.cat((memory, context), -1)
        # Feed it to RNN
        # s_i = f(y_{i-1}, c_{i}, s_{i-1})
        rnn_output = self.rnn_cell(rnn_input, rnn_state)
        # Alignment
        # (batch, max_time)
        # e_{ij} = a(s_{i-1}, h_j)
        alignment = self.alignment_model(annotations, rnn_output)
        # TODO: needs recheck.
        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)
        # Normalize context weight
        alignment = F.softmax(alignment, dim=-1)
        # Attention context vector
        # (batch, 1, dim)
        # c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
        context = torch.bmm(alignment.unsqueeze(1), annotations)
        context = context.squeeze(1)
        return rnn_output, context, alignment
