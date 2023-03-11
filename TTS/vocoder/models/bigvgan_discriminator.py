import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Conv2d
from torch.nn.utils import spectral_norm, weight_norm

from TTS.utils.audio.torch_transforms import TorchSTFT
from TTS.vocoder.models.hifigan_discriminator import MultiPeriodDiscriminator
from TTS.vocoder.models.univnet_discriminator import MultiResSpecDiscriminator
from TTS.vocoder.utils.generic_utils import get_padding

LRELU_SLOPE = 0.1


class BigvganDiscriminator(nn.Module):
    """BigVgan discriminator wrapping one MultiResolutionDiscriminator and a MultiPeriodDiscriminator.

    ::
        waveform -> MultiResolutionDiscriminator() -> scores_mrd, feats_mrd
               |--> MultiPeriodDiscriminator() -> scores_mpd, feats_mpd ^ --> append() -> scores, feats

    Args:
        use_spectral_norm (bool): if `True` swith to spectral norm instead of weight norm.
    """

    def __init__(self, use_spectral_norm=False):
        super().__init__()
        # for res in resolutions:
        self.msd = MultiResSpecDiscriminator()
        self.mpd = MultiPeriodDiscriminator(use_spectral_norm=use_spectral_norm)

    def forward(self, x):
        """
        Args:
            x (Tensor): input waveform.

        Returns:
            List[Tensor]: discriminator scores.
            List[List[Tensor]]: list of list of features from each layers of each discriminator.
        """
        scores, feats = self.mpd(x)
        scores_, feats_ = self.msd(x)
        return scores + scores_, feats + feats_
