from typing import Dict, Union

import torch
from torch import nn
from torch.nn import functional as F

from TTS.utils.audio.torch_transforms import TorchSTFT
from TTS.vocoder.utils.distribution import discretized_mix_logistic_loss, gaussian_loss

#################################
# GENERATOR LOSSES
#################################


class STFTLoss(nn.Module):
    """STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, n_fft, hop_length, win_length):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.win_length = win_length
        self.stft = TorchSTFT(n_fft, hop_length, win_length)

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        # spectral convergence loss
        loss_sc = torch.norm(y_M - y_hat_M, p="fro") / torch.norm(y_M, p="fro")
        return loss_mag, loss_sc


class MultiScaleSTFTLoss(torch.nn.Module):
    """Multi-scale STFT loss. Input generate and real waveforms are converted
    to spectrograms compared with L1 and Spectral convergence losses.
    It is from ParallelWaveGAN paper https://arxiv.org/pdf/1910.11480.pdf"""

    def __init__(self, n_ffts=(1024, 2048, 512), hop_lengths=(120, 240, 50), win_lengths=(600, 1200, 240)):
        super().__init__()
        self.loss_funcs = torch.nn.ModuleList()
        for n_fft, hop_length, win_length in zip(n_ffts, hop_lengths, win_lengths):
            self.loss_funcs.append(STFTLoss(n_fft, hop_length, win_length))

    def forward(self, y_hat, y):
        N = len(self.loss_funcs)
        loss_sc = 0
        loss_mag = 0
        for f in self.loss_funcs:
            lm, lsc = f(y_hat, y)
            loss_mag += lm
            loss_sc += lsc
        loss_sc /= N
        loss_mag /= N
        return loss_mag, loss_sc


class L1SpecLoss(nn.Module):
    """L1 Loss over Spectrograms as described in HiFiGAN paper https://arxiv.org/pdf/2010.05646.pdf"""

    def __init__(
        self, sample_rate, n_fft, hop_length, win_length, mel_fmin=None, mel_fmax=None, n_mels=None, use_mel=True
    ):
        super().__init__()
        self.use_mel = use_mel
        self.stft = TorchSTFT(
            n_fft,
            hop_length,
            win_length,
            sample_rate=sample_rate,
            mel_fmin=mel_fmin,
            mel_fmax=mel_fmax,
            n_mels=n_mels,
            use_mel=use_mel,
        )

    def forward(self, y_hat, y):
        y_hat_M = self.stft(y_hat)
        y_M = self.stft(y)
        # magnitude loss
        loss_mag = F.l1_loss(torch.log(y_M), torch.log(y_hat_M))
        return loss_mag


class MultiScaleSubbandSTFTLoss(MultiScaleSTFTLoss):
    """Multiscale STFT loss for multi band model outputs.
    From MultiBand-MelGAN paper https://arxiv.org/abs/2005.05106"""

    # pylint: disable=no-self-use
    def forward(self, y_hat, y):
        y_hat = y_hat.view(-1, 1, y_hat.shape[2])
        y = y.view(-1, 1, y.shape[2])
        return super().forward(y_hat.squeeze(1), y.squeeze(1))


class MSEGLoss(nn.Module):
    """Mean Squared Generator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_real):
        loss_fake = F.mse_loss(score_real, score_real.new_ones(score_real.shape))
        return loss_fake


class HingeGLoss(nn.Module):
    """Hinge Discriminator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_real):
        # TODO: this might be wrong
        loss_fake = torch.mean(F.relu(1.0 - score_real))
        return loss_fake


##################################
# DISCRIMINATOR LOSSES
##################################


class MSEDLoss(nn.Module):
    """Mean Squared Discriminator Loss"""

    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.MSELoss()

    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real):
        loss_real = self.loss_func(score_real, score_real.new_ones(score_real.shape))
        loss_fake = self.loss_func(score_fake, score_fake.new_zeros(score_fake.shape))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class HingeDLoss(nn.Module):
    """Hinge Discriminator Loss"""

    # pylint: disable=no-self-use
    def forward(self, score_fake, score_real):
        loss_real = torch.mean(F.relu(1.0 - score_real))
        loss_fake = torch.mean(F.relu(1.0 + score_fake))
        loss_d = loss_real + loss_fake
        return loss_d, loss_real, loss_fake


class MelganFeatureLoss(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()
        self.loss_func = nn.L1Loss()

    # pylint: disable=no-self-use
    def forward(self, fake_feats, real_feats):
        loss_feats = 0
        num_feats = 0
        for idx, _ in enumerate(fake_feats):
            for fake_feat, real_feat in zip(fake_feats[idx], real_feats[idx]):
                loss_feats += self.loss_func(fake_feat, real_feat)
                num_feats += 1
        loss_feats = loss_feats / num_feats
        return loss_feats


#####################################
# LOSS WRAPPERS
#####################################


def _apply_G_adv_loss(scores_fake, loss_func):
    """Compute G adversarial loss function
    and normalize values"""
    adv_loss = 0
    if isinstance(scores_fake, list):
        for score_fake in scores_fake:
            fake_loss = loss_func(score_fake)
            adv_loss += fake_loss
        adv_loss /= len(scores_fake)
    else:
        fake_loss = loss_func(scores_fake)
        adv_loss = fake_loss
    return adv_loss


def _apply_D_loss(scores_fake, scores_real, loss_func):
    """Compute D loss func and normalize loss values"""
    loss = 0
    real_loss = 0
    fake_loss = 0
    if isinstance(scores_fake, list):
        # multi-scale loss
        for score_fake, score_real in zip(scores_fake, scores_real):
            total_loss, real_loss_, fake_loss_ = loss_func(score_fake=score_fake, score_real=score_real)
            loss += total_loss
            real_loss += real_loss_
            fake_loss += fake_loss_
        # normalize loss values with number of scales (discriminators)
        loss /= len(scores_fake)
        real_loss /= len(scores_real)
        fake_loss /= len(scores_fake)
    else:
        # single scale loss
        total_loss, real_loss, fake_loss = loss_func(scores_fake, scores_real)
        loss = total_loss
    return loss, real_loss, fake_loss


##################################
# MODEL LOSSES
##################################


class GeneratorLoss(nn.Module):
    """Generator Loss Wrapper. Based on model configuration it sets a right set of loss functions and computes
    losses. It allows to experiment with different combinations of loss functions with different models by just
    changing configurations.

    Args:
        C (AttrDict): model configuration.
    """

    def __init__(self, C):
        super().__init__()
        assert not (
            C.use_mse_gan_loss and C.use_hinge_gan_loss
        ), " [!] Cannot use HingeGANLoss and MSEGANLoss together."

        self.use_stft_loss = C.use_stft_loss if "use_stft_loss" in C else False
        self.use_subband_stft_loss = C.use_subband_stft_loss if "use_subband_stft_loss" in C else False
        self.use_mse_gan_loss = C.use_mse_gan_loss if "use_mse_gan_loss" in C else False
        self.use_hinge_gan_loss = C.use_hinge_gan_loss if "use_hinge_gan_loss" in C else False
        self.use_feat_match_loss = C.use_feat_match_loss if "use_feat_match_loss" in C else False
        self.use_l1_spec_loss = C.use_l1_spec_loss if "use_l1_spec_loss" in C else False

        self.stft_loss_weight = C.stft_loss_weight if "stft_loss_weight" in C else 0.0
        self.subband_stft_loss_weight = C.subband_stft_loss_weight if "subband_stft_loss_weight" in C else 0.0
        self.mse_gan_loss_weight = C.mse_G_loss_weight if "mse_G_loss_weight" in C else 0.0
        self.hinge_gan_loss_weight = C.hinge_G_loss_weight if "hinde_G_loss_weight" in C else 0.0
        self.feat_match_loss_weight = C.feat_match_loss_weight if "feat_match_loss_weight" in C else 0.0
        self.l1_spec_loss_weight = C.l1_spec_loss_weight if "l1_spec_loss_weight" in C else 0.0

        if C.use_stft_loss:
            self.stft_loss = MultiScaleSTFTLoss(**C.stft_loss_params)
        if C.use_subband_stft_loss:
            self.subband_stft_loss = MultiScaleSubbandSTFTLoss(**C.subband_stft_loss_params)
        if C.use_mse_gan_loss:
            self.mse_loss = MSEGLoss()
        if C.use_hinge_gan_loss:
            self.hinge_loss = HingeGLoss()
        if C.use_feat_match_loss:
            self.feat_match_loss = MelganFeatureLoss()
        if C.use_l1_spec_loss:
            assert C.audio["sample_rate"] == C.l1_spec_loss_params["sample_rate"]
            self.l1_spec_loss = L1SpecLoss(**C.l1_spec_loss_params)

    def forward(
        self, y_hat=None, y=None, scores_fake=None, feats_fake=None, feats_real=None, y_hat_sub=None, y_sub=None
    ):
        gen_loss = 0
        adv_loss = 0
        return_dict = {}

        # STFT Loss
        if self.use_stft_loss:
            stft_loss_mg, stft_loss_sc = self.stft_loss(y_hat[:, :, : y.size(2)].squeeze(1), y.squeeze(1))
            return_dict["G_stft_loss_mg"] = stft_loss_mg
            return_dict["G_stft_loss_sc"] = stft_loss_sc
            gen_loss = gen_loss + self.stft_loss_weight * (stft_loss_mg + stft_loss_sc)

        # L1 Spec loss
        if self.use_l1_spec_loss:
            l1_spec_loss = self.l1_spec_loss(y_hat, y)
            return_dict["G_l1_spec_loss"] = l1_spec_loss
            gen_loss = gen_loss + self.l1_spec_loss_weight * l1_spec_loss

        # subband STFT Loss
        if self.use_subband_stft_loss:
            subband_stft_loss_mg, subband_stft_loss_sc = self.subband_stft_loss(y_hat_sub, y_sub)
            return_dict["G_subband_stft_loss_mg"] = subband_stft_loss_mg
            return_dict["G_subband_stft_loss_sc"] = subband_stft_loss_sc
            gen_loss = gen_loss + self.subband_stft_loss_weight * (subband_stft_loss_mg + subband_stft_loss_sc)

        # multiscale MSE adversarial loss
        if self.use_mse_gan_loss and scores_fake is not None:
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mse_loss)
            return_dict["G_mse_fake_loss"] = mse_fake_loss
            adv_loss = adv_loss + self.mse_gan_loss_weight * mse_fake_loss

        # multiscale Hinge adversarial loss
        if self.use_hinge_gan_loss and not scores_fake is not None:
            hinge_fake_loss = _apply_G_adv_loss(scores_fake, self.hinge_loss)
            return_dict["G_hinge_fake_loss"] = hinge_fake_loss
            adv_loss = adv_loss + self.hinge_gan_loss_weight * hinge_fake_loss

        # Feature Matching Loss
        if self.use_feat_match_loss and not feats_fake is None:
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            return_dict["G_feat_match_loss"] = feat_match_loss
            adv_loss = adv_loss + self.feat_match_loss_weight * feat_match_loss
        return_dict["loss"] = gen_loss + adv_loss
        return_dict["G_gen_loss"] = gen_loss
        return_dict["G_adv_loss"] = adv_loss
        return return_dict


class DiscriminatorLoss(nn.Module):
    """Like ```GeneratorLoss```"""

    def __init__(self, C):
        super().__init__()
        assert not (
            C.use_mse_gan_loss and C.use_hinge_gan_loss
        ), " [!] Cannot use HingeGANLoss and MSEGANLoss together."

        self.use_mse_gan_loss = C.use_mse_gan_loss
        self.use_hinge_gan_loss = C.use_hinge_gan_loss

        if C.use_mse_gan_loss:
            self.mse_loss = MSEDLoss()
        if C.use_hinge_gan_loss:
            self.hinge_loss = HingeDLoss()

    def forward(self, scores_fake, scores_real):
        loss = 0
        return_dict = {}

        if self.use_mse_gan_loss:
            mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake, scores_real=scores_real, loss_func=self.mse_loss
            )
            return_dict["D_mse_gan_loss"] = mse_D_loss
            return_dict["D_mse_gan_real_loss"] = mse_D_real_loss
            return_dict["D_mse_gan_fake_loss"] = mse_D_fake_loss
            loss += mse_D_loss

        if self.use_hinge_gan_loss:
            hinge_D_loss, hinge_D_real_loss, hinge_D_fake_loss = _apply_D_loss(
                scores_fake=scores_fake, scores_real=scores_real, loss_func=self.hinge_loss
            )
            return_dict["D_hinge_gan_loss"] = hinge_D_loss
            return_dict["D_hinge_gan_real_loss"] = hinge_D_real_loss
            return_dict["D_hinge_gan_fake_loss"] = hinge_D_fake_loss
            loss += hinge_D_loss

        return_dict["loss"] = loss
        return return_dict


class WaveRNNLoss(nn.Module):
    def __init__(self, wave_rnn_mode: Union[str, int]):
        super().__init__()
        if wave_rnn_mode == "mold":
            self.loss_func = discretized_mix_logistic_loss
        elif wave_rnn_mode == "gauss":
            self.loss_func = gaussian_loss
        elif isinstance(wave_rnn_mode, int):
            self.loss_func = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(" [!] Unknown mode for Wavernn.")

    def forward(self, y_hat, y) -> Dict:
        loss = self.loss_func(y_hat, y)
        return {"loss": loss}
