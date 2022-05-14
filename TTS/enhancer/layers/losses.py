import torch

from TTS.tts.layers.losses import L1LossMasked
from TTS.vocoder.layers.losses import (
    L1SpecLoss,
    MelganFeatureLoss,
    MSEDLoss,
    MSEGLoss,
    MultiScaleSTFTLoss,
    _apply_D_loss,
    _apply_G_adv_loss,
)


class BWEGeneratorLoss(torch.nn.Module):
    def __init__(self, n_scale_STFTLoss=4, sr=48000, n_fft=2024, hop_length=512, n_mels=128):
        super().__init__()
        self.l1_masked = L1LossMasked(False)
        self.stft_loss = MultiScaleSTFTLoss(
            n_ffts=tuple(512 * 2**i for i in range(n_scale_STFTLoss)),
            hop_lengths=tuple(int(512 * 2**i / 4) for i in range(n_scale_STFTLoss)),
            win_lengths=tuple(512 * 2**i for i in range(n_scale_STFTLoss)),
        )
        self.mel_loss = L1SpecLoss(sr, n_fft, hop_length, n_fft, mel_fmin=0, mel_fmax=sr//2, n_mels=n_mels, use_mel=True)
        self.feat_match_loss = MelganFeatureLoss()
        self.gen_gan_loss = MSEGLoss()
        self.pred_l1_loss = torch.nn.L1Loss()
        self.pred_l2_loss = torch.nn.MSELoss()

    def forward(
            self,
            y_hat,
            y,
            y_hat_postnet=None,
            lens=None,
            scores_fake=None,
            feats_fake=None,
            feats_real=None,
            mfcc=None,
            mfcc_hat=None
        ):
        return_dict = {}
        if lens is None:
            lens = torch.IntTensor([y.size(1)]).to(y.device)

        # Waveform loss
        return_dict["G_l1_wavform"] = self.l1_masked(y_hat, y, lens)
        return_dict["loss"] = return_dict["G_l1_wavform"] * 10

        # Spectrogram losses
        return_dict["G_stft_loss_mg"], return_dict["G_stft_loss_sc"] = self.stft_loss(y_hat.squeeze(1), y.squeeze(1))
        return_dict["loss"] += (return_dict["G_stft_loss_mg"] + return_dict["G_stft_loss_sc"]) * 1

        if y_hat_postnet is not None:
            return_dict["G_postnet_l1_wavform"] = self.l1_masked(y_hat_postnet, y, lens)
            return_dict["loss"] += return_dict["G_postnet_l1_wavform"] * 10
            return_dict["G_postnet_stft_loss_mg"], return_dict["G_postnet_stft_loss_sc"] = self.stft_loss(
                y_hat_postnet.squeeze(1),
                y.squeeze(1)
            )
            return_dict["loss"] += (return_dict["G_postnet_stft_loss_mg"] + return_dict["G_postnet_stft_loss_sc"]) * 1

        if mfcc is not None and mfcc_hat is not None:
            return_dict["pred_l1_loss"] = self.pred_l1_loss(y_hat, y)
            return_dict["pred_l2_loss"] = self.pred_l2_loss(y_hat, y)
            return_dict["loss"] += (return_dict["pred_l1_loss"] + return_dict["pred_l2_loss"]) * 1

        if scores_fake is not None and feats_fake is not None and feats_real is not None:
            # Feature matching loss
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            return_dict["G_feat_match"] = feat_match_loss
            return_dict["loss"] += feat_match_loss * 108

            # MSE adversarial loss
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.gen_gan_loss)
            return_dict["G_mse_fake"] = mse_fake_loss
            return_dict["loss"] += return_dict["G_mse_fake"] * 2

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict


class BWEDiscriminatorLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.disc_gan_loss = MSEDLoss()

    def forward(self, scores_fake, scores_real):
        return_dict = {}
        mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
            scores_fake=scores_fake, scores_real=scores_real, loss_func=self.disc_gan_loss
        )
        return_dict["D_mse"] = mse_D_loss
        return_dict["D_mse_real"] = mse_D_real_loss
        return_dict["D_mse_fake"] = mse_D_fake_loss
        return_dict["loss"] = return_dict["D_mse"]
        return return_dict
