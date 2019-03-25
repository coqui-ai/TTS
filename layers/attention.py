import torch
from torch import nn
from torch.nn import functional as F
from utils.generic_utils import sequence_mask


class BahdanauAttention(nn.Module):
    def __init__(self, annot_dim, query_dim, attn_dim):
        super(BahdanauAttention, self).__init__()
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, annots, query):
        """
        Shapes:
            - annots: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        # (batch, 1, dim)
        processed_query = self.query_layer(query)
        processed_annots = self.annot_layer(annots)
        # (batch, max_time, 1)
        alignment = self.v(torch.tanh(processed_query + processed_annots))
        # (batch, max_time)
        return alignment.squeeze(-1)


class LocationSensitiveAttention(nn.Module):
    """Location sensitive attention following
    https://arxiv.org/pdf/1506.07503.pdf"""

    def __init__(self,
                 annot_dim,
                 query_dim,
                 attn_dim,
                 kernel_size=31,
                 filters=32):
        super(LocationSensitiveAttention, self).__init__()
        self.kernel_size = kernel_size
        self.filters = filters
        padding = [(kernel_size - 1) // 2, (kernel_size - 1) // 2]
        self.loc_conv = nn.Sequential(
            nn.ConstantPad1d(padding, 0),
            nn.Conv1d(
                2,
                filters,
                kernel_size=kernel_size,
                stride=1,
                padding=0,
                bias=False))
        self.loc_linear = nn.Linear(filters, attn_dim, bias=True)
        self.query_layer = nn.Linear(query_dim, attn_dim, bias=True)
        self.annot_layer = nn.Linear(annot_dim, attn_dim, bias=True)
        self.v = nn.Linear(attn_dim, 1, bias=False)
        self.processed_annots = None
        # self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(
            self.loc_linear.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.query_layer.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.annot_layer.weight,
            gain=torch.nn.init.calculate_gain('tanh'))
        torch.nn.init.xavier_uniform_(
            self.v.weight,
            gain=torch.nn.init.calculate_gain('linear'))

    def reset(self):
        self.processed_annots = None

    def forward(self, annot, query, loc):
        """
        Shapes:
            - annot: (batch, max_time, dim)
            - query: (batch, 1, dim) or (batch, dim)
            - loc: (batch, 2, max_time)
        """
        if query.dim() == 2:
            # insert time-axis for broadcasting
            query = query.unsqueeze(1)
        processed_loc = self.loc_linear(self.loc_conv(loc).transpose(1, 2))
        processed_query = self.query_layer(query)
        # cache annots
        if self.processed_annots is None:
            self.processed_annots = self.annot_layer(annot)
        alignment = self.v(
            torch.tanh(processed_query + self.processed_annots + processed_loc))
        del processed_loc
        del processed_query
        # (batch, max_time)
        return alignment.squeeze(-1)


class AttentionRNNCell(nn.Module):
    def __init__(self, out_dim, rnn_dim, annot_dim, memory_dim, align_model, windowing=False, norm="sigmoid"):
        r"""
        General Attention RNN wrapper

        Args:
            out_dim (int): context vector feature dimension.
            rnn_dim (int): rnn hidden state dimension.
            annot_dim (int): annotation vector feature dimension.
            memory_dim (int): memory vector (decoder output) feature dimension.
            align_model (str): 'b' for Bahdanau, 'ls' Location Sensitive alignment.
            windowing (bool): attention windowing forcing monotonic attention.
                It is only active in eval mode.
            norm (str): norm method to compute alignment weights.
        """
        super(AttentionRNNCell, self).__init__()
        self.align_model = align_model
        self.rnn_cell = nn.GRUCell(annot_dim + memory_dim, rnn_dim)
        self.windowing = windowing
        if self.windowing:
            self.win_back = 3
            self.win_front = 6
            self.win_idx = None
        self.norm = norm
        if align_model == 'b':
            self.alignment_model = BahdanauAttention(annot_dim, rnn_dim,
                                                     out_dim)
        if align_model == 'ls':
            self.alignment_model = LocationSensitiveAttention(
                annot_dim, rnn_dim, out_dim)
        else:
            raise RuntimeError(" Wrong alignment model name: {}. Use\
                'b' (Bahdanau) or 'ls' (Location Sensitive).".format(
                align_model))

    def forward(self, memory, context, rnn_state, annots, atten, mask, t):
        """
        Shapes:
            - memory: (batch, 1, dim) or (batch, dim)
            - context: (batch, dim)
            - rnn_state: (batch, out_dim)
            - annots: (batch, max_time, annot_dim)
            - atten: (batch, 2, max_time)
            - mask: (batch,)
        """
        if t == 0:
            self.alignment_model.reset()
            self.win_idx = 0
        rnn_output = self.rnn_cell(torch.cat((memory, context), -1), rnn_state)
        if self.align_model is 'b':
            alignment = self.alignment_model(annots, rnn_output)
        else:
            alignment = self.alignment_model(annots, rnn_output, atten)
        if mask is not None:
            mask = mask.view(memory.size(0), -1)
            alignment.masked_fill_(1 - mask, -float("inf"))
        # Windowing
        if not self.training and self.windowing:
            back_win = self.win_idx - self.win_back
            front_win = self.win_idx + self.win_front
            if back_win > 0:
                alignment[:, :back_win] = -float("inf")
            if front_win < memory.shape[1]:
                alignment[:, front_win:] = -float("inf")
            # Update the window
            self.win_idx = torch.argmax(alignment,1).long()[0].item()
        if self.norm == "softmax":
            alignment = torch.softmax(alignment, dim=-1)
        elif self.norm == "sigmoid":
            alignment = torch.sigmoid(alignment) / torch.sigmoid(alignment).sum(dim=1).unsqueeze(1)
        else:
            raise RuntimeError("Unknown value for attention norm type")
        context = torch.bmm(alignment.unsqueeze(1), annots)
        context = context.squeeze(1)
        return rnn_output, context, alignment
