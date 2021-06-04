import torch
from torch.nn import functional as F
from torch.nn import Sequential, Embedding, Linear


from torch._six import container_abcs

from itertools import repeat
from typing import List
from torch import Tensor
from typing import Sequence, Tuple


def _ntuple(n):
    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse

_pair = _ntuple(2)

class _ConstantPadNd(torch.nn.Module):
    __constants__ = ['padding', 'value']
    value: float
    padding: Sequence[int]

    def __init__(self, value: float) -> None:
        super(_ConstantPadNd, self).__init__()
        self.value = value

    def forward(self, input: Tensor) -> Tensor:
        return F.pad(input, self.padding, 'constant', self.value)

    def extra_repr(self) -> str:
        return 'padding={}, value={}'.format(self.padding, self.value)

class ConstantPad1d(_ConstantPadNd):
    r"""Pads the input tensor boundaries with a constant value.

    For `N`-dimensional padding, use :func:`torch.nn.functional.pad()`.

    Args:
        padding (int, tuple): the size of the padding. If is `int`, uses the same
            padding in both boundaries. If a 2-`tuple`, uses
            (:math:`\text{padding\_left}`, :math:`\text{padding\_right}`)

    Shape:
        - Input: :math:`(N, C, W_{in})`
        - Output: :math:`(N, C, W_{out})` where

          :math:`W_{out} = W_{in} + \text{padding\_left} + \text{padding\_right}`

    Examples::

        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 4)
        >>> input
        tensor([[[-1.0491, -0.7152, -0.0749,  0.8530],
                 [-1.3287,  1.8966,  0.1466, -0.2771]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000, -1.0491, -0.7152, -0.0749,  0.8530,  3.5000,
                   3.5000],
                 [ 3.5000,  3.5000, -1.3287,  1.8966,  0.1466, -0.2771,  3.5000,
                   3.5000]]])
        >>> m = nn.ConstantPad1d(2, 3.5)
        >>> input = torch.randn(1, 2, 3)
        >>> input
        tensor([[[ 1.6616,  1.4523, -1.1255],
                 [-3.6372,  0.1182, -1.8652]]])
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000,  3.5000],
                 [ 3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000,  3.5000]]])
        >>> # using different paddings for different sides
        >>> m = nn.ConstantPad1d((3, 1), 3.5)
        >>> m(input)
        tensor([[[ 3.5000,  3.5000,  3.5000,  1.6616,  1.4523, -1.1255,  3.5000],
                 [ 3.5000,  3.5000,  3.5000, -3.6372,  0.1182, -1.8652,  3.5000]]])

    """
    padding: Tuple[int, int]

    def __init__(self, padding, value: float):
        super(ConstantPad1d, self).__init__(value)
        self.padding = _pair(padding)

class Conv1dGenerated(torch.nn.Module):
    """One dimensional convolution with generated weights (each group has separate weights).
    
    Arguments:
        embedding_dim -- size of the meta embedding (should be language embedding)
        bottleneck_dim -- size of the generating embedding
        see torch.nn.Conv1d
    """

    def __init__(self, embedding_dim, bottleneck_dim, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(Conv1dGenerated, self).__init__()

        self._in_channels = in_channels
        self._out_channels = out_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding
        self._dilation = dilation
        self._groups = groups

        # in_channels and out_channels is divisible by groups
        # tf.nn.functional.conv1d accepts weights of shape [out_channels, in_channels // groups, kernel] 

        self._bottleneck = Linear(embedding_dim, bottleneck_dim)
        self._kernel = Linear(bottleneck_dim, out_channels // groups * in_channels // groups * kernel_size)
        self._bias = Linear(bottleneck_dim, out_channels // groups) if bias else None
        
    def forward(self, generator_embedding, x):

        assert generator_embedding.shape[0] == self._groups, ('Number of groups of a convolutional layer must match the number of generators.')

        e = self._bottleneck(generator_embedding)
        kernel = self._kernel(e).view(self._out_channels, self._in_channels // self._groups, self._kernel_size)
        bias = self._bias(e).view(self._out_channels) if self._bias else None

        return F.conv1d(x, kernel, bias, self._stride, self._padding, self._dilation, self._groups)


class BatchNorm1dGenerated(torch.nn.Module):
    """One dimensional batch normalization with generated weights (each group has separate parameters).
    
    Arguments:
        embedding_dim -- size of the meta embedding (should be language embedding)
        bottleneck_dim -- size of the generating embedding
        see torch.nn.BatchNorm1d
    Keyword arguments:
        groups -- number of groups with separate weights
    """

    def __init__(self, embedding_dim, bottleneck_dim, num_features, groups=1, eps=1e-8, momentum=0.1):
        super(BatchNorm1dGenerated, self).__init__()

        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        self.register_buffer('num_batches_tracked', torch.tensor(0, dtype=torch.long))

        self._num_features = num_features // groups
        self._eps = eps
        self._momentum = momentum
        self._groups = groups
        
        self._bottleneck = Linear(embedding_dim, bottleneck_dim)
        self._affine = Linear(bottleneck_dim, self._num_features + self._num_features)
   
    def forward(self, generator_embedding, x):

        assert generator_embedding.shape[0] == self._groups, (
            'Number of groups of a batchnorm layer must match the number of generators.')

        if self._momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self._momentum

        e = self._bottleneck(generator_embedding)
        affine = self._affine(e)
        scale = affine[:, :self._num_features].contiguous().view(-1)
        bias = affine[:, self._num_features:].contiguous().view(-1)

        if self.training:
            if self.num_batches_tracked is not None:
                self.num_batches_tracked += 1
                if self._momentum is None:
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else: 
                    exponential_average_factor = self._momentum

        return F.batch_norm(
            x, self.running_mean, self.running_var, scale, bias, 
            self.training, exponential_average_factor, self._eps)

class ConvBlockGenerated(torch.nn.Module):
    """One dimensional convolution with generated weights and with batchnorm and dropout, expected channel-first input.
    
    Arguments:
        embedding_dim -- size of the meta embedding
        bottleneck_dim -- size of the generating layer
        input_channels -- number if input channels
        output_channels -- number of output channels
        kernel -- convolution kernel size ('same' padding is used)
    Keyword arguments:
        dropout (default: 0.0) -- dropout rate to be aplied after the block
        activation (default 'identity') -- name of the activation function applied after batchnorm
        dilation (default: 1) -- dilation of the inner convolution
        groups (default: 1) -- number of groups of the inner convolution
        batch_norm (default: True) -- set False to disable batch normalization
    """

    def __init__(self, embedding_dim, bottleneck_dim, input_channels, output_channels, kernel,
                 dropout=0.0, activation='identity', dilation=1, groups=1, batch_norm=True):
        super(ConvBlockGenerated, self).__init__()

        self._groups = groups
        
        p = (kernel-1) * dilation // 2 
        padding = p if kernel % 2 != 0 else (p, p+1)
        
        self._padding = ConstantPad1d(padding, 0.0)
        self._convolution = Conv1dGenerated(embedding_dim, bottleneck_dim, input_channels, output_channels, kernel, 
                                     padding=0, dilation=dilation, groups=groups, bias=(not batch_norm))
        self._regularizer = BatchNorm1dGenerated(embedding_dim, bottleneck_dim, output_channels, groups=groups) if batch_norm else None
        self._activation = Sequential(
            torch.nn.ReLU(),
            torch.nn.Dropout()
        )

    def forward(self, x):
        e, x = x
        x = self._padding(x)
        x = self._convolution(e, x)
        if self._regularizer is not None:
            x = self._regularizer(e, x)
        x = self._activation(x)
        return e, x

class HighwayConvBlockGenerated(ConvBlockGenerated):
    """Gated 1D covolution with generated weights.
    
    Arguments:
        embedding_dim -- size of the meta embedding
        bottleneck_dim -- size of the generating layer
        see ConvBlockGenerated
    """

    def __init__(self, embedding_dim, bottleneck_dim, input_channels, output_channels, kernel, 
                 dropout=0.0, activation='identity', dilation=1, groups=1, batch_norm=True):
        super(HighwayConvBlockGenerated, self).__init__(embedding_dim, bottleneck_dim, input_channels, 2*output_channels, kernel, 
                                                        dropout, activation, dilation, groups, batch_norm)
        self._gate = torch.nn.Sigmoid()

    def forward(self, x):
        e, x = x
        _, h = super(HighwayConvBlockGenerated, self).forward((e, x))
        chunks = torch.chunk(h, 2 * self._groups, 1)
        h1 = torch.cat(chunks[0::2], 1)
        h2 = torch.cat(chunks[1::2], 1)
        p = self._gate(h1)
        return e, h2 * p + x * (1.0 - p)



class GeneratedConvolutionalEncoder(torch.nn.Module):
    """Convolutional encoder (possibly multi-lingual) with weights generated by another network.
    
    Arguments:
        see ConvolutionalEncoder
        embedding_dim -- size of the generator embedding (should be language embedding)
        bottleneck_dim -- size of the generating layer
    Keyword arguments:
        see ConvolutionalEncoder
    """

    def __init__(self, input_dim, output_dim, dropout=0.05, embedding_dim=10, bottleneck_dim=4, groups=1):
        super(GeneratedConvolutionalEncoder, self).__init__()
        
        self._groups = groups
        self._input_dim = input_dim
        self._output_dim = output_dim
        
        input_dim *= groups
        output_dim *= groups
        
        layers = [ConvBlockGenerated(embedding_dim, bottleneck_dim, input_dim, output_dim, 1,
                                     dropout=dropout, activation='relu', groups=groups),
                  ConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 1,
                                     dropout=dropout, groups=groups)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 3, 
                                            dropout=dropout, dilation=3**i, groups=groups) for i in range(4)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 3,
                                            dropout=dropout, dilation=3**i, groups=groups) for i in range(4)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 3,
                                            dropout=dropout, dilation=1, groups=groups) for _ in range(2)] + \
                 [HighwayConvBlockGenerated(embedding_dim, bottleneck_dim, output_dim, output_dim, 1,
                                            dropout=dropout, dilation=1, groups=groups) for _ in range(2)]
        
        self._layers = Sequential(*layers)
        self._embedding = Embedding(groups, embedding_dim)

    def forward(self, x, x_lenghts=None, language_ids=None):

        # language_ids is specified during inference with batch size 1, so we need to 
        # expand the single language to create complete groups (all langs. in parallel)
        if language_ids is not None and language_ids.shape[0] == 1:
            x = x.expand((self._groups, -1, -1))

        # create generator embeddings for all groups
        e = self._embedding(torch.arange(self._groups-1, -1, -1, device=x.device)) # all embeddings in the reversed order

        bs = x.shape[0]
        x = x.transpose(1, 2)
        x = x.reshape(bs // self._groups, self._groups * self._input_dim, -1)   
        _, x = self._layers((e, x))
        x = x.reshape(bs, self._output_dim, -1)
        x = x.transpose(1, 2)

        if language_ids is not None and language_ids.shape[0] == 1:
            xr = torch.zeros(1, x.shape[1], x.shape[2], device=x.device)
            x_langs_normed = language_ids / language_ids.sum(2, keepdim=True)[0]
            for l in range(self._groups):
                w = x_langs_normed[0,:,l].reshape(-1,1)
                xr[0] += w * x[l]
            x = xr

        return x