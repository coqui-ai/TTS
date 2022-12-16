import torch
from torch.autograd import Function

'''
    Gradient Reversal Layer implementation (based on: https://github.com/jvanvugt/pytorch-domain-adaptation/blob/master/utils.py)

    Gradient Reversal Layer from:
    Unsupervised Domain Adaptation by Backpropagation (Ganin & Lempitsky, 2015)
    Forward pass is the identity function. In the backward pass,
    the upstream gradients are multiplied by -lambda (i.e. gradient is reversed)

'''

# Function
class GradientReversalLayer_(Function):
    @staticmethod
    def forward(ctx, input, lambda_):
        ctx.lambda_ = lambda_
        return input

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.lambda_ 
        lambda_ = grad_output.new_tensor(lambda_)
        grad_input = -lambda_ * grad_output
        return grad_input, None

# Class to use in models
class GradientReversalLayer(torch.nn.Module):
    def __init__(self, lambda_ = 1):
        super(GradientReversalLayer, self).__init__()
        self.lambda_ = lambda_ 

    def forward(self, x):
        return GradientReversalLayer_.apply(x, self.lambda_)