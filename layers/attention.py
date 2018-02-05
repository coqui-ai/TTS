import torch
from torch.autograd import Variable
from torch import nn
from torch.nn import functional as F


class BahdanauAttention(nn.Module):
    def __init__(self, dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(dim, dim, bias=False)
        self.tanh = nn.Tanh()
        self.v = nn.Linear(dim, 1, bias=False)

    def forward(self, query, processed_inputs):
        """
        Args:
            query: (batch, 1, dim) or (batch, dim)
            processed_inputs: (batch, max_time, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)

        # (batch, max_time, 1)
        alignment = self.v(self.tanh(processed_query + processed_inputs))

        # (batch, max_time)
        return alignment.squeeze(-1)


def get_mask_from_lengths(memory, memory_lengths):
    """Get mask tensor from list of length

    Args:
        memory: (batch, max_time, dim)
        memory_lengths: array like
    """
    mask = memory.data.new(memory.size(0), memory.size(1)).byte().zero_()
    for idx, l in enumerate(memory_lengths):
        mask[idx][:l] = 1
    return ~mask


class AttentionWrapper(nn.Module):
    def __init__(self, rnn_cell, alignment_model,
                 score_mask_value=-float("inf")):
        super(AttentionWrapper, self).__init__()
        self.rnn_cell = rnn_cell
        self.alignment_model = alignment_model
        self.score_mask_value = score_mask_value

    def forward(self, query, context_vec, cell_state, memory,
                processed_inputs=None, mask=None, memory_lengths=None):

        if processed_inputs is None:
            processed_inputs = memory

        if memory_lengths is not None and mask is None:
            mask = get_mask_from_lengths(memory, memory_lengths)

        # Concat input query and previous context_vec context
        import ipdb
        ipdb.set_trace()
        cell_input = torch.cat((query, context_vec), -1)

        # Feed it to RNN
        cell_output = self.rnn_cell(cell_input, cell_state)

        # Alignment
        # (batch, max_time)
        alignment = self.alignment_model(cell_output, processed_inputs)

        if mask is not None:
            mask = mask.view(query.size(0), -1)
            alignment.data.masked_fill_(mask, self.score_mask_value)

        # Normalize context_vec weight
        alignment = F.softmax(alignment, dim=-1)

        # Attention context vector
        # (batch, 1, dim)
        context_vec = torch.bmm(alignment.unsqueeze(1), memory)

        # (batch, dim)
        context_vec = context_vec.squeeze(1)

        return cell_output, context_vec, alignment


