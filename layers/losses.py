import torch
from torch.nn import functional
from torch import nn
from utils.generic_utils import sequence_mask


class L1LossMasked(nn.Module):
    def __init__(self):
        super(L1LossMasked, self).__init__()

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        # mask: (batch, max_len, 1)
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2).float()
        mask = mask.expand_as(input)
        loss = functional.l1_loss(
            input * mask, target * mask, reduction="sum")
        loss = loss / mask.sum()
        return loss


class MSELossMasked(nn.Module):
    def __init__(self):
        super(MSELossMasked, self).__init__()

    def forward(self, input, target, length):
        """
        Args:
            input: A Variable containing a FloatTensor of size
                (batch, max_len, dim) which contains the
                unnormalized probability for each class.
            target: A Variable containing a LongTensor of size
                (batch, max_len, dim) which contains the index of the true
                class for each corresponding step.
            length: A Variable containing a LongTensor of size (batch,)
                which contains the length of each data in a batch.
        Returns:
            loss: An average loss value masked by the length.
        """
        input = input.contiguous()
        target = target.contiguous()

        # logits_flat: (batch * max_len, dim)
        input = input.view(-1, input.shape[-1])
        # target_flat: (batch * max_len, dim)
        target_flat = target.view(-1, target.shape[-1])
        # losses_flat: (batch * max_len, dim)
        losses_flat = functional.mse_loss(
            input, target_flat, size_average=False, reduce=False)
        # losses: (batch, max_len, dim)
        losses = losses_flat.view(*target.size())

        # mask: (batch, max_len, 1)
        mask = sequence_mask(
            sequence_length=length, max_len=target.size(1)).unsqueeze(2)
        losses = losses * mask.float()
        loss = losses.sum() / (length.float().sum() * float(target.shape[2]))
        return loss
