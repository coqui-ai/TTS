# Adopted from https://github.com/photosynthesis-team/piq

from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss


def _reduce(x: torch.Tensor, reduction: str = "mean") -> torch.Tensor:
    r"""Reduce input in batch dimension if needed.
    Args:
        x: Tensor with shape (N, *).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default: ``'mean'``
    """
    if reduction == "none":
        return x
    if reduction == "mean":
        return x.mean(dim=0)
    if reduction == "sum":
        return x.sum(dim=0)
    raise ValueError("Unknown reduction. Expected one of {'none', 'mean', 'sum'}")


def _validate_input(
    tensors: List[torch.Tensor],
    dim_range: Tuple[int, int] = (0, -1),
    data_range: Tuple[float, float] = (0.0, -1.0),
    # size_dim_range: Tuple[float, float] = (0., -1.),
    size_range: Optional[Tuple[int, int]] = None,
) -> None:
    r"""Check that input(-s)  satisfies the requirements
    Args:
        tensors: Tensors to check
        dim_range: Allowed number of dimensions. (min, max)
        data_range: Allowed range of values in tensors. (min, max)
        size_range: Dimensions to include in size comparison. (start_dim, end_dim + 1)
    """

    if not __debug__:
        return

    x = tensors[0]

    for t in tensors:
        assert torch.is_tensor(t), f"Expected torch.Tensor, got {type(t)}"
        assert t.device == x.device, f"Expected tensors to be on {x.device}, got {t.device}"

        if size_range is None:
            assert t.size() == x.size(), f"Expected tensors with same size, got {t.size()} and {x.size()}"
        else:
            assert (
                t.size()[size_range[0] : size_range[1]] == x.size()[size_range[0] : size_range[1]]
            ), f"Expected tensors with same size at given dimensions, got {t.size()} and {x.size()}"

        if dim_range[0] == dim_range[1]:
            assert t.dim() == dim_range[0], f"Expected number of dimensions to be {dim_range[0]}, got {t.dim()}"
        elif dim_range[0] < dim_range[1]:
            assert (
                dim_range[0] <= t.dim() <= dim_range[1]
            ), f"Expected number of dimensions to be between {dim_range[0]} and {dim_range[1]}, got {t.dim()}"

        if data_range[0] < data_range[1]:
            assert data_range[0] <= t.min(), f"Expected values to be greater or equal to {data_range[0]}, got {t.min()}"
            assert t.max() <= data_range[1], f"Expected values to be lower or equal to {data_range[1]}, got {t.max()}"


def gaussian_filter(kernel_size: int, sigma: float) -> torch.Tensor:
    r"""Returns 2D Gaussian kernel N(0,`sigma`^2)
    Args:
        size: Size of the kernel
        sigma: Std of the distribution
    Returns:
        gaussian_kernel: Tensor with shape (1, kernel_size, kernel_size)
    """
    coords = torch.arange(kernel_size, dtype=torch.float32)
    coords -= (kernel_size - 1) / 2.0

    g = coords**2
    g = (-(g.unsqueeze(0) + g.unsqueeze(1)) / (2 * sigma**2)).exp()

    g /= g.sum()
    return g.unsqueeze(0)


def ssim(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel_size: int = 11,
    kernel_sigma: float = 1.5,
    data_range: Union[int, float] = 1.0,
    reduction: str = "mean",
    full: bool = False,
    downsample: bool = True,
    k1: float = 0.01,
    k2: float = 0.03,
) -> List[torch.Tensor]:
    r"""Interface of Structural Similarity (SSIM) index.
    Inputs supposed to be in range ``[0, data_range]``.
    To match performance with skimage and tensorflow set ``'downsample' = True``.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
        kernel_size: The side-length of the sliding window used in comparison. Must be an odd value.
        kernel_sigma: Sigma of normal distribution.
        data_range: Maximum value range of images (usually 1.0 or 255).
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        full: Return cs map or not.
        downsample: Perform average pool before SSIM computation. Default: True
        k1: Algorithm parameter, K1 (small constant).
        k2: Algorithm parameter, K2 (small constant).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Value of Structural Similarity (SSIM) index. In case of 5D input tensors, complex value is returned
        as a tensor of size 2.

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI: `10.1109/TIP.2003.819861`
    """
    assert kernel_size % 2 == 1, f"Kernel size must be odd, got [{kernel_size}]"
    _validate_input([x, y], dim_range=(4, 5), data_range=(0, data_range))

    x = x / float(data_range)
    y = y / float(data_range)

    # Averagepool image if the size is large enough
    f = max(1, round(min(x.size()[-2:]) / 256))
    if (f > 1) and downsample:
        x = F.avg_pool2d(x, kernel_size=f)
        y = F.avg_pool2d(y, kernel_size=f)

    kernel = gaussian_filter(kernel_size, kernel_sigma).repeat(x.size(1), 1, 1, 1).to(y)
    _compute_ssim_per_channel = _ssim_per_channel_complex if x.dim() == 5 else _ssim_per_channel
    ssim_map, cs_map = _compute_ssim_per_channel(x=x, y=y, kernel=kernel, k1=k1, k2=k2)
    ssim_val = ssim_map.mean(1)
    cs = cs_map.mean(1)

    ssim_val = _reduce(ssim_val, reduction)
    cs = _reduce(cs, reduction)

    if full:
        return [ssim_val, cs]

    return ssim_val


class SSIMLoss(_Loss):
    r"""Creates a criterion that measures the structural similarity index error between
    each element in the input :math:`x` and target :math:`y`.

    To match performance with skimage and tensorflow set ``'downsample' = True``.

    The unreduced (i.e. with :attr:`reduction` set to ``'none'``) loss can be described as:

    .. math::
        SSIM = \{ssim_1,\dots,ssim_{N \times C}\}\\
        ssim_{l}(x, y) = \frac{(2 \mu_x \mu_y + c_1) (2 \sigma_{xy} + c_2)}
        {(\mu_x^2 +\mu_y^2 + c_1)(\sigma_x^2 +\sigma_y^2 + c_2)},

    where :math:`N` is the batch size, `C` is the channel size. If :attr:`reduction` is not ``'none'``
    (default ``'mean'``), then:

    .. math::
        SSIMLoss(x, y) =
        \begin{cases}
            \operatorname{mean}(1 - SSIM), &  \text{if reduction} = \text{'mean';}\\
            \operatorname{sum}(1 - SSIM),  &  \text{if reduction} = \text{'sum'.}
        \end{cases}

    :math:`x` and :math:`y` are tensors of arbitrary shapes with a total
    of :math:`n` elements each.

    The sum operation still operates over all the elements, and divides by :math:`n`.
    The division by :math:`n` can be avoided if one sets ``reduction = 'sum'``.
    In case of 5D input tensors, complex value is returned as a tensor of size 2.

    Args:
        kernel_size: By default, the mean and covariance of a pixel is obtained
            by convolution with given filter_size.
        kernel_sigma: Standard deviation for Gaussian kernel.
        k1: Coefficient related to c1 in the above equation.
        k2: Coefficient related to c2 in the above equation.
        downsample: Perform average pool before SSIM computation. Default: True
        reduction: Specifies the reduction type:
            ``'none'`` | ``'mean'`` | ``'sum'``. Default:``'mean'``
        data_range: Maximum value range of images (usually 1.0 or 255).

    Examples:
        >>> loss = SSIMLoss()
        >>> x = torch.rand(3, 3, 256, 256, requires_grad=True)
        >>> y = torch.rand(3, 3, 256, 256)
        >>> output = loss(x, y)
        >>> output.backward()

    References:
        Wang, Z., Bovik, A. C., Sheikh, H. R., & Simoncelli, E. P. (2004).
        Image quality assessment: From error visibility to structural similarity.
        IEEE Transactions on Image Processing, 13, 600-612.
        https://ece.uwaterloo.ca/~z70wang/publications/ssim.pdf,
        DOI:`10.1109/TIP.2003.819861`
    """
    __constants__ = ["kernel_size", "k1", "k2", "sigma", "kernel", "reduction"]

    def __init__(
        self,
        kernel_size: int = 11,
        kernel_sigma: float = 1.5,
        k1: float = 0.01,
        k2: float = 0.03,
        downsample: bool = True,
        reduction: str = "mean",
        data_range: Union[int, float] = 1.0,
    ) -> None:
        super().__init__()

        # Generic loss parameters.
        self.reduction = reduction

        # Loss-specific parameters.
        self.kernel_size = kernel_size

        # This check might look redundant because kernel size is checked within the ssim function anyway.
        # However, this check allows to fail fast when the loss is being initialised and training has not been started.
        assert kernel_size % 2 == 1, f"Kernel size must be odd, got [{kernel_size}]"
        self.kernel_sigma = kernel_sigma
        self.k1 = k1
        self.k2 = k2
        self.downsample = downsample
        self.data_range = data_range

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r"""Computation of Structural Similarity (SSIM) index as a loss function.

        Args:
            x: An input tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.
            y: A target tensor. Shape :math:`(N, C, H, W)` or :math:`(N, C, H, W, 2)`.

        Returns:
            Value of SSIM loss to be minimized, i.e ``1 - ssim`` in [0, 1] range. In case of 5D input tensors,
            complex value is returned as a tensor of size 2.
        """

        score = ssim(
            x=x,
            y=y,
            kernel_size=self.kernel_size,
            kernel_sigma=self.kernel_sigma,
            downsample=self.downsample,
            data_range=self.data_range,
            reduction=self.reduction,
            full=False,
            k1=self.k1,
            k2=self.k2,
        )
        return torch.ones_like(score) - score


def _ssim_per_channel(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W)`.
        y: A target tensor. Shape :math:`(N, C, H, W)`.
        kernel: 2D Gaussian kernel.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Structural Similarity (SSIM) index.
    """
    if x.size(-1) < kernel.size(-1) or x.size(-2) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. Input size: {x.size()}. "
            f"Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2
    n_channels = x.size(1)
    mu_x = F.conv2d(x, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu_y = F.conv2d(y, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu_xx = mu_x**2
    mu_yy = mu_y**2
    mu_xy = mu_x * mu_y

    sigma_xx = F.conv2d(x**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xx
    sigma_yy = F.conv2d(y**2, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_yy
    sigma_xy = F.conv2d(x * y, weight=kernel, stride=1, padding=0, groups=n_channels) - mu_xy

    # Contrast sensitivity (CS) with alpha = beta = gamma = 1.
    cs = (2.0 * sigma_xy + c2) / (sigma_xx + sigma_yy + c2)

    # Structural similarity (SSIM)
    ss = (2.0 * mu_xy + c1) / (mu_xx + mu_yy + c1) * cs

    ssim_val = ss.mean(dim=(-1, -2))
    cs = cs.mean(dim=(-1, -2))
    return ssim_val, cs


def _ssim_per_channel_complex(
    x: torch.Tensor,
    y: torch.Tensor,
    kernel: torch.Tensor,
    k1: float = 0.01,
    k2: float = 0.03,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    r"""Calculate Structural Similarity (SSIM) index for Complex X and Y per channel.

    Args:
        x: An input tensor. Shape :math:`(N, C, H, W, 2)`.
        y: A target tensor. Shape :math:`(N, C, H, W, 2)`.
        kernel: 2-D gauss kernel.
        k1: Algorithm parameter, K1 (small constant, see [1]).
        k2: Algorithm parameter, K2 (small constant, see [1]).
            Try a larger K2 constant (e.g. 0.4) if you get a negative or NaN results.

    Returns:
        Full Value of Complex Structural Similarity (SSIM) index.
    """
    n_channels = x.size(1)
    if x.size(-2) < kernel.size(-1) or x.size(-3) < kernel.size(-2):
        raise ValueError(
            f"Kernel size can't be greater than actual input size. Input size: {x.size()}. "
            f"Kernel size: {kernel.size()}"
        )

    c1 = k1**2
    c2 = k2**2

    x_real = x[..., 0]
    x_imag = x[..., 1]
    y_real = y[..., 0]
    y_imag = y[..., 1]

    mu1_real = F.conv2d(x_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu1_imag = F.conv2d(x_imag, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_real = F.conv2d(y_real, weight=kernel, stride=1, padding=0, groups=n_channels)
    mu2_imag = F.conv2d(y_imag, weight=kernel, stride=1, padding=0, groups=n_channels)

    mu1_sq = mu1_real.pow(2) + mu1_imag.pow(2)
    mu2_sq = mu2_real.pow(2) + mu2_imag.pow(2)
    mu1_mu2_real = mu1_real * mu2_real - mu1_imag * mu2_imag
    mu1_mu2_imag = mu1_real * mu2_imag + mu1_imag * mu2_real

    compensation = 1.0

    x_sq = x_real.pow(2) + x_imag.pow(2)
    y_sq = y_real.pow(2) + y_imag.pow(2)
    x_y_real = x_real * y_real - x_imag * y_imag
    x_y_imag = x_real * y_imag + x_imag * y_real

    sigma1_sq = F.conv2d(x_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_sq
    sigma2_sq = F.conv2d(y_sq, weight=kernel, stride=1, padding=0, groups=n_channels) - mu2_sq
    sigma12_real = F.conv2d(x_y_real, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_real
    sigma12_imag = F.conv2d(x_y_imag, weight=kernel, stride=1, padding=0, groups=n_channels) - mu1_mu2_imag
    sigma12 = torch.stack((sigma12_imag, sigma12_real), dim=-1)
    mu1_mu2 = torch.stack((mu1_mu2_real, mu1_mu2_imag), dim=-1)
    # Set alpha = beta = gamma = 1.
    cs_map = (sigma12 * 2 + c2 * compensation) / (sigma1_sq.unsqueeze(-1) + sigma2_sq.unsqueeze(-1) + c2 * compensation)
    ssim_map = (mu1_mu2 * 2 + c1 * compensation) / (mu1_sq.unsqueeze(-1) + mu2_sq.unsqueeze(-1) + c1 * compensation)
    ssim_map = ssim_map * cs_map

    ssim_val = ssim_map.mean(dim=(-2, -3))
    cs = cs_map.mean(dim=(-2, -3))

    return ssim_val, cs
