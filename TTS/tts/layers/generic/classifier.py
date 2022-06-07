import torch
from torch import nn


# pylint: disable=W0223
class GradientReversalFunction(torch.autograd.Function):
    """Revert gradient without any further input modification.
    Adapted from: https://github.com/Tomiinek/Multilingual_Text_to_Speech/"""

    @staticmethod
    def forward(ctx, x, l, c):
        ctx.l = l
        ctx.c = c
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clamp(-ctx.c, ctx.c)
        return ctx.l * grad_output.neg(), None, None


class ReversalClassifier(nn.Module):
    """Adversarial classifier with an optional gradient reversal layer.
    Adapted from: https://github.com/Tomiinek/Multilingual_Text_to_Speech/

    Args:
        in_channels (int): Number of input tensor channels.
        out_channels (int): Number of output tensor channels (Number of classes).
        hidden_channels (int): Number of hidden channels.
        gradient_clipping_bound (float): Maximal value of the gradient which flows from this module. Default: 0.25
        scale_factor (float): Scale multiplier of the reversed gradientts. Default: 1.0
        reversal (bool): If True reversal the gradients. Default: True
    """

    def __init__(
        self, in_channels, out_channels, hidden_channels, gradient_clipping_bounds=0.25, scale_factor=1.0, reversal=True
    ):
        super().__init__()
        self.reversal = reversal
        self._lambda = scale_factor
        self._clipping = gradient_clipping_bounds
        self._out_channels = out_channels
        self._classifier = nn.Sequential(
            nn.Linear(in_channels, hidden_channels), nn.ReLU(), nn.Linear(hidden_channels, out_channels)
        )

    def forward(self, x, labels, x_mask=None):
        if self.reversal:
            x = GradientReversalFunction.apply(x, self._lambda, self._clipping)
        x = self._classifier(x)
        loss = self.loss(labels, x, x_mask)
        return x, loss

    @staticmethod
    def loss(labels, predictions, x_mask=None, ignore_index=-100):
        if x_mask is None:
            x_mask = torch.Tensor([predictions.size(1)]).repeat(predictions.size(0)).int().to(predictions.device)
            ml = torch.max(x_mask)
            input_mask = torch.arange(ml, device=predictions.device)[None, :] < x_mask[:, None]
        else:
            input_mask = x_mask.squeeze().bool()
        target = labels.repeat(input_mask.size(-1), 1).transpose(0, 1).int().long()
        target[~input_mask] = ignore_index
        return nn.functional.cross_entropy(predictions.transpose(1, 2), target, ignore_index=ignore_index)
