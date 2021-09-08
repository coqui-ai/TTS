import torch
import numpy as np


class StandardScaler:
    """StandardScaler for mean-std normalization with the given mean and std values.
    """
    def __init__(self, mean:np.ndarray=None, std:np.ndarray=None) -> None:
        self.mean_ = mean
        self.std_ = std

    def set_stats(self, mean, scale):
        self.mean_ = mean
        self.scale_ = scale

    def reset_stats(self):
        delattr(self, "mean_")
        delattr(self, "scale_")

    def transform(self, X):
        X = np.asarray(X)
        X -= self.mean_
        X /= self.scale_
        return X

    def inverse_transform(self, X):
        X = np.asarray(X)
        X *= self.scale_
        X += self.mean_
        return X


# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    mask = seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)
    return mask


def segment(x: torch.tensor, segment_indices: torch.tensor, segment_size=4):
    """Segment each sample in a batch based on the provided segment indices

    Args:
        x (torch.tensor): Input tensor.
        segment_indices (torch.tensor): Segment indices.
        segment_size (int): Expected output segment size.
    """
    segments = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        segments[i] = x[i, :, index_start:index_end]
    return segments


def rand_segments(x: torch.tensor, x_lengths: torch.tensor = None, segment_size=4):
    """Create random segments based on the input lengths.

    Args:
        x (torch.tensor): Input tensor.
        x_lengths (torch.tensor): Input lengths.
        segment_size (int): Expected output segment size.

    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    B, _, T = x.size()
    if x_lengths is None:
        x_lengths = T
    max_idxs = x_lengths - segment_size + 1
    assert all(max_idxs > 0), " [!] At least one sample is shorter than the segment size."
    segment_indices = (torch.rand([B]).type_as(x) * max_idxs).long()
    ret = segment(x, segment_indices, segment_size)
    return ret, segment_indices

def average_over_durations(values, durs):
    """Average values over durations.

    Shapes:
        - values: :math:`[B, 1, T_de]`
        - durs: :math:`[B, T_en]`
        - avg: :math:`[B, 1, T_en]`
    """
    durs_cums_ends = torch.cumsum(durs, dim=1).long()
    durs_cums_starts = torch.nn.functional.pad(durs_cums_ends[:, :-1], (1, 0))
    values_nonzero_cums = torch.nn.functional.pad(torch.cumsum(values != 0.0, dim=2), (1, 0))
    values_cums = torch.nn.functional.pad(torch.cumsum(values, dim=2), (1, 0))

    bs, l = durs_cums_ends.size()
    n_formants = values.size(1)
    dcs = durs_cums_starts[:, None, :].expand(bs, n_formants, l)
    dce = durs_cums_ends[:, None, :].expand(bs, n_formants, l)

    values_sums = (torch.gather(values_cums, 2, dce) - torch.gather(values_cums, 2, dcs)).float()
    values_nelems = (torch.gather(values_nonzero_cums, 2, dce) - torch.gather(values_nonzero_cums, 2, dcs)).float()

    avg = torch.where(values_nelems == 0.0, values_nelems, values_sums / values_nelems)
    return avg