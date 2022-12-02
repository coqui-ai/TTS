import torch


def get_spec_from_most_probable_state(log_alpha_scaled, means):
    """Get the most probable state means from the log_alpha_scaled.

    Args:
        log_alpha_scaled (torch.Tensor): Log alpha scaled values.
            - Shape: :math:`(T, N)`
        means (torch.Tensor): Means of the states.
            - Shape: :math:`(N, T, D_out)`
    """
    max_state_numbers = torch.max(log_alpha_scaled, dim=1)[1]
    max_len = means.shape[0]
    n_mel_channels = means.shape[2]
    max_state_numbers = max_state_numbers.unsqueeze(1).unsqueeze(1).expand(max_len, 1, n_mel_channels)
    means = torch.gather(means, 1, max_state_numbers).squeeze(1)
    return means
