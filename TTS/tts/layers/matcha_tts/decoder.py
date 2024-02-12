import torch
from torch import nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_min = 1e-5

    def forward(self, x_1, mean, mask):
        """
        Shapes:
            - x_1:  :math:`[B, C, T]`
            - mean: :math:`[B, C ,T]`
            - mask: :math:`[B, 1, T]`
        """
        t = torch.rand([x_1.size(0), 1, 1], device=x_1.device, dtype=x_1.dtype)
        x_0 = torch.randn_like(x_1)
        x_t = (1 - (1 - self.sigma_min) * t) * x_0 + t * x_1
        u_t = x_1 - (1 - self.sigma_min) * x_0
        v_t = torch.randn_like(u_t)
        loss = F.mse_loss(v_t, u_t, reduction="sum") / (torch.sum(mask) * u_t.shape[1])
        return loss
