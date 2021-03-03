import torch
from torch import nn
from torch.nn import functional as F


class Linear(nn.Module):
    """Linear layer with a specific initialization.

    Args:
        in_features (int): number of channels in the input tensor.
        out_features (int): number of channels in the output tensor.
        bias (bool, optional): enable/disable bias in the layer. Defaults to True.
        init_gain (str, optional): method to compute the gain in the weight initializtion based on the nonlinear activation used afterwards. Defaults to 'linear'.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_gain='linear'):
        super(Linear, self).__init__()
        self.linear_layer = torch.nn.Linear(
            in_features, out_features, bias=bias)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        return self.linear_layer(x)


class LinearBN(nn.Module):
    """Linear layer with Batch Normalization.

    x -> linear -> BN -> o

    Args:
        in_features (int): number of channels in the input tensor.
        out_features (int ): number of channels in the output tensor.
        bias (bool, optional): enable/disable bias in the linear layer. Defaults to True.
        init_gain (str, optional): method to set the gain for weight initialization. Defaults to 'linear'.
    """
    def __init__(self,
                 in_features,
                 out_features,
                 bias=True,
                 init_gain='linear'):
        super(LinearBN, self).__init__()
        self.linear_layer = torch.nn.Linear(
            in_features, out_features, bias=bias)
        self.batch_normalization = nn.BatchNorm1d(out_features, momentum=0.1, eps=1e-5)
        self._init_w(init_gain)

    def _init_w(self, init_gain):
        torch.nn.init.xavier_uniform_(
            self.linear_layer.weight,
            gain=torch.nn.init.calculate_gain(init_gain))

    def forward(self, x):
        """
        Shapes:
            x: [T, B, C] or [B, C]
        """
        out = self.linear_layer(x)
        if len(out.shape) == 3:
            out = out.permute(1, 2, 0)
        out = self.batch_normalization(out)
        if len(out.shape) == 3:
            out = out.permute(2, 0, 1)
        return out


class Prenet(nn.Module):
    """Tacotron specific Prenet with an optional Batch Normalization.

    Note:
        Prenet with BN improves the model performance significantly especially
    if it is enabled after learning a diagonal attention alignment with the original
    prenet. However, if the target dataset is high quality then it also works from
    the start. It is also suggested to disable dropout if BN is in use.

        prenet_type == "original"
            x -> [linear -> ReLU -> Dropout]xN -> o

        prenet_type == "bn"
            x -> [linear -> BN -> ReLU -> Dropout]xN -> o

    Args:
        in_features (int): number of channels in the input tensor and the inner layers.
        prenet_type (str, optional): prenet type "original" or "bn". Defaults to "original".
        prenet_dropout (bool, optional): dropout rate. Defaults to True.
        out_features (list, optional): List of output channels for each prenet block.
            It also defines number of the prenet blocks based on the length of argument list.
            Defaults to [256, 256].
        bias (bool, optional): enable/disable bias in prenet linear layers. Defaults to True.
    """
    # pylint: disable=dangerous-default-value
    def __init__(self,
                 in_features,
                 prenet_type="original",
                 prenet_dropout=True,
                 out_features=[256, 256],
                 bias=True):
        super(Prenet, self).__init__()
        self.prenet_type = prenet_type
        self.prenet_dropout = prenet_dropout
        in_features = [in_features] + out_features[:-1]
        if prenet_type == "bn":
            self.linear_layers = nn.ModuleList([
                LinearBN(in_size, out_size, bias=bias)
                for (in_size, out_size) in zip(in_features, out_features)
            ])
        elif prenet_type == "original":
            self.linear_layers = nn.ModuleList([
                Linear(in_size, out_size, bias=bias)
                for (in_size, out_size) in zip(in_features, out_features)
            ])

    def forward(self, x):
        for linear in self.linear_layers:
            if self.prenet_dropout:
                x = F.dropout(F.relu(linear(x)), p=0.5, training=self.training)
            else:
                x = F.relu(linear(x))
        return x
