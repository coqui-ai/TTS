from typing import Tuple

import torch
from torch import nn


class AlignmentNetwork(torch.nn.Module):
    """Aligner Network for learning alignment between the input text and the model output with Gaussian Attention.

    ::

        query -> conv1d -> relu -> conv1d -> relu -> conv1d -> L2_dist -> softmax -> alignment
        key   -> conv1d -> relu -> conv1d -----------------------^

    Args:
        in_query_channels (int): Number of channels in the query network. Defaults to 80.
        in_key_channels (int): Number of channels in the key network. Defaults to 512.
        attn_channels (int): Number of inner channels in the attention layers. Defaults to 80.
        temperature (float): Temperature for the softmax. Defaults to 0.0005.
    """

    def __init__(
        self,
        in_query_channels=80,
        in_key_channels=512,
        attn_channels=80,
        temperature=0.0005,
    ):
        super().__init__()
        self.temperature = temperature
        self.softmax = torch.nn.Softmax(dim=3)
        self.log_softmax = torch.nn.LogSoftmax(dim=3)

        self.key_layer = nn.Sequential(
            nn.Conv1d(
                in_key_channels,
                in_key_channels * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            nn.Conv1d(in_key_channels * 2, attn_channels, kernel_size=1, padding=0, bias=True),
        )

        self.query_layer = nn.Sequential(
            nn.Conv1d(
                in_query_channels,
                in_query_channels * 2,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            torch.nn.ReLU(),
            nn.Conv1d(in_query_channels * 2, in_query_channels, kernel_size=1, padding=0, bias=True),
            torch.nn.ReLU(),
            nn.Conv1d(in_query_channels, attn_channels, kernel_size=1, padding=0, bias=True),
        )

        self.init_layers()

    def init_layers(self):
        torch.nn.init.xavier_uniform_(self.key_layer[0].weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self.key_layer[2].weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self.query_layer[0].weight, gain=torch.nn.init.calculate_gain("relu"))
        torch.nn.init.xavier_uniform_(self.query_layer[2].weight, gain=torch.nn.init.calculate_gain("linear"))
        torch.nn.init.xavier_uniform_(self.query_layer[4].weight, gain=torch.nn.init.calculate_gain("linear"))

    def forward(
        self, queries: torch.tensor, keys: torch.tensor, mask: torch.tensor = None, attn_prior: torch.tensor = None
    ) -> Tuple[torch.tensor, torch.tensor]:
        """Forward pass of the aligner encoder.
        Shapes:
            - queries: :math:`[B, C, T_de]`
            - keys: :math:`[B, C_emb, T_en]`
            - mask: :math:`[B, T_de]`
        Output:
            attn (torch.tensor): :math:`[B, 1, T_en, T_de]` soft attention mask.
            attn_logp (torch.tensor): :math:`[ßB, 1, T_en , T_de]` log probabilities.
        """
        key_out = self.key_layer(keys)
        query_out = self.query_layer(queries)
        attn_factor = (query_out[:, :, :, None] - key_out[:, :, None]) ** 2
        attn_logp = -self.temperature * attn_factor.sum(1, keepdim=True)
        if attn_prior is not None:
            attn_logp = self.log_softmax(attn_logp) + torch.log(attn_prior[:, None] + 1e-8)

        if mask is not None:
            attn_logp.data.masked_fill_(~mask.bool().unsqueeze(2), -float("inf"))

        attn = self.softmax(attn_logp)
        return attn, attn_logp
