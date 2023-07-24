from typing import List, Tuple, Union

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F

from TTS.tts.layers.delightful_tts.conformer import ConformerMultiHeadedSelfAttention
from TTS.tts.layers.delightful_tts.conv_layers import CoordConv1d
from TTS.tts.layers.delightful_tts.networks import STL


def get_mask_from_lengths(lengths: torch.Tensor) -> torch.Tensor:
    batch_size = lengths.shape[0]
    max_len = torch.max(lengths).item()
    ids = torch.arange(0, max_len, device=lengths.device).unsqueeze(0).expand(batch_size, -1)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)
    return mask


def stride_lens(lens: torch.Tensor, stride: int = 2) -> torch.Tensor:
    return torch.ceil(lens / stride).int()


class ReferenceEncoder(nn.Module):
    """
    Referance encoder for utterance and phoneme prosody encoders. Reference encoder
    made up of convolution and RNN layers.

    Args:
        num_mels (int): Number of mel frames to produce.
        ref_enc_filters (list[int]): List of channel sizes for encoder layers.
        ref_enc_size (int): Size of the kernel for the conv layers.
        ref_enc_strides (List[int]): List of strides to use for conv layers.
        ref_enc_gru_size (int): Number of hidden features for the gated recurrent unit.

    Inputs: inputs, mask
        - **inputs** (batch, dim, time): Tensor containing mel vector
        - **lengths** (batch): Tensor containing the mel lengths.
    Returns:
        - **outputs** (batch, time, dim): Tensor produced by Reference Encoder.
    """

    def __init__(
        self,
        num_mels: int,
        ref_enc_filters: List[Union[int, int, int, int, int, int]],
        ref_enc_size: int,
        ref_enc_strides: List[Union[int, int, int, int, int]],
        ref_enc_gru_size: int,
    ):
        super().__init__()

        n_mel_channels = num_mels
        self.n_mel_channels = n_mel_channels
        K = len(ref_enc_filters)
        filters = [self.n_mel_channels] + ref_enc_filters
        strides = [1] + ref_enc_strides
        # Use CoordConv at the first layer to better preserve positional information: https://arxiv.org/pdf/1811.02122.pdf
        convs = [
            CoordConv1d(
                in_channels=filters[0],
                out_channels=filters[0 + 1],
                kernel_size=ref_enc_size,
                stride=strides[0],
                padding=ref_enc_size // 2,
                with_r=True,
            )
        ]
        convs2 = [
            nn.Conv1d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=ref_enc_size,
                stride=strides[i],
                padding=ref_enc_size // 2,
            )
            for i in range(1, K)
        ]
        convs.extend(convs2)
        self.convs = nn.ModuleList(convs)

        self.norms = nn.ModuleList([nn.InstanceNorm1d(num_features=ref_enc_filters[i], affine=True) for i in range(K)])

        self.gru = nn.GRU(
            input_size=ref_enc_filters[-1],
            hidden_size=ref_enc_gru_size,
            batch_first=True,
        )

    def forward(self, x: torch.Tensor, mel_lens: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        inputs --- [N,  n_mels, timesteps]
        outputs --- [N, E//2]
        """

        mel_masks = get_mask_from_lengths(mel_lens).unsqueeze(1)
        x = x.masked_fill(mel_masks, 0)
        for conv, norm in zip(self.convs, self.norms):
            x = conv(x)
            x = F.leaky_relu(x, 0.3)  # [N, 128, Ty//2^K, n_mels//2^K]
            x = norm(x)

        for _ in range(2):
            mel_lens = stride_lens(mel_lens)

        mel_masks = get_mask_from_lengths(mel_lens)

        x = x.masked_fill(mel_masks.unsqueeze(1), 0)
        x = x.permute((0, 2, 1))
        x = torch.nn.utils.rnn.pack_padded_sequence(x, mel_lens.cpu().int(), batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        x, memory = self.gru(x)  # memory --- [N, Ty, E//2], out --- [1, N, E//2]
        x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        return x, memory, mel_masks

    def calculate_channels(  # pylint: disable=no-self-use
        self, L: int, kernel_size: int, stride: int, pad: int, n_convs: int
    ) -> int:
        for _ in range(n_convs):
            L = (L - kernel_size + 2 * pad) // stride + 1
        return L


class UtteranceLevelProsodyEncoder(nn.Module):
    def __init__(
        self,
        num_mels: int,
        ref_enc_filters: List[Union[int, int, int, int, int, int]],
        ref_enc_size: int,
        ref_enc_strides: List[Union[int, int, int, int, int]],
        ref_enc_gru_size: int,
        dropout: float,
        n_hidden: int,
        bottleneck_size_u: int,
        token_num: int,
    ):
        """
        Encoder to extract prosody from utterance. it is made up of a reference encoder
        with a couple of linear layers and style token layer with dropout.

        Args:
            num_mels (int): Number of mel frames to produce.
            ref_enc_filters (list[int]): List of channel sizes for ref encoder layers.
            ref_enc_size (int): Size of the kernel for the ref encoder conv layers.
            ref_enc_strides (List[int]): List of strides to use for teh ref encoder conv layers.
            ref_enc_gru_size (int): Number of hidden features for the gated recurrent unit.
            dropout (float): Probability of dropout.
            n_hidden (int): Size of hidden layers.
            bottleneck_size_u (int): Size of the bottle neck layer.

        Inputs: inputs, mask
            - **inputs** (batch, dim, time): Tensor containing mel vector
            - **lengths** (batch): Tensor containing the mel lengths.
        Returns:
            - **outputs** (batch, 1, dim): Tensor produced by Utterance Level Prosody Encoder.
        """
        super().__init__()

        self.E = n_hidden
        self.d_q = self.d_k = n_hidden
        bottleneck_size = bottleneck_size_u

        self.encoder = ReferenceEncoder(
            ref_enc_filters=ref_enc_filters,
            ref_enc_gru_size=ref_enc_gru_size,
            ref_enc_size=ref_enc_size,
            ref_enc_strides=ref_enc_strides,
            num_mels=num_mels,
        )
        self.encoder_prj = nn.Linear(ref_enc_gru_size, self.E // 2)
        self.stl = STL(n_hidden=n_hidden, token_num=token_num)
        self.encoder_bottleneck = nn.Linear(self.E, bottleneck_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, mels: torch.Tensor, mel_lens: torch.Tensor) -> torch.Tensor:
        """
        Shapes:
            mels: :math: `[B, C, T]`
            mel_lens: :math: `[B]`

        out --- [N, seq_len, E]
        """
        _, embedded_prosody, _ = self.encoder(mels, mel_lens)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        # Style Token
        out = self.encoder_bottleneck(self.stl(embedded_prosody))
        out = self.dropout(out)

        out = out.view((-1, 1, out.shape[3]))
        return out


class PhonemeLevelProsodyEncoder(nn.Module):
    def __init__(
        self,
        num_mels: int,
        ref_enc_filters: List[Union[int, int, int, int, int, int]],
        ref_enc_size: int,
        ref_enc_strides: List[Union[int, int, int, int, int]],
        ref_enc_gru_size: int,
        dropout: float,
        n_hidden: int,
        n_heads: int,
        bottleneck_size_p: int,
    ):
        super().__init__()

        self.E = n_hidden
        self.d_q = self.d_k = n_hidden
        bottleneck_size = bottleneck_size_p

        self.encoder = ReferenceEncoder(
            ref_enc_filters=ref_enc_filters,
            ref_enc_gru_size=ref_enc_gru_size,
            ref_enc_size=ref_enc_size,
            ref_enc_strides=ref_enc_strides,
            num_mels=num_mels,
        )
        self.encoder_prj = nn.Linear(ref_enc_gru_size, n_hidden)
        self.attention = ConformerMultiHeadedSelfAttention(
            d_model=n_hidden,
            num_heads=n_heads,
            dropout_p=dropout,
        )
        self.encoder_bottleneck = nn.Linear(n_hidden, bottleneck_size)

    def forward(
        self,
        x: torch.Tensor,
        src_mask: torch.Tensor,
        mels: torch.Tensor,
        mel_lens: torch.Tensor,
        encoding: torch.Tensor,
    ) -> torch.Tensor:
        """
        x --- [N, seq_len, encoder_embedding_dim]
        mels --- [N, Ty/r, n_mels*r], r=1
        out --- [N, seq_len, bottleneck_size]
        attn --- [N, seq_len, ref_len], Ty/r = ref_len
        """
        embedded_prosody, _, mel_masks = self.encoder(mels, mel_lens)

        # Bottleneck
        embedded_prosody = self.encoder_prj(embedded_prosody)

        attn_mask = mel_masks.view((mel_masks.shape[0], 1, 1, -1))
        x, _ = self.attention(
            query=x,
            key=embedded_prosody,
            value=embedded_prosody,
            mask=attn_mask,
            encoding=encoding,
        )
        x = self.encoder_bottleneck(x)
        x = x.masked_fill(src_mask.unsqueeze(-1), 0.0)
        return x
