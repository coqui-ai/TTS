from typing import Type
import torch
from TTS.utils.audio import TorchSTFT
from TTS.tts.layers.losses import L1LossMasked
from TTS.vocoder.layers.losses import MelganFeatureLoss, MSEGLoss, MSEDLoss, _apply_G_adv_loss, _apply_D_loss, MultiScaleSTFTLoss, L1SpecLoss

class BWEGeneratorLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_masked = L1LossMasked(False)
        self.stft_loss = MultiScaleSTFTLoss(
            n_ffts=tuple(512*2**i for i in range(4)), 
            hop_lengths=tuple(int(512*2**i/4) for i in range(4)), 
            win_lengths=tuple(512*2**i for i in range(4))
        )
        self.mel_loss = L1SpecLoss(48000, 2024, 512, 2024, mel_fmin=0, mel_fmax=24000, n_mels=128, use_mel=True)
        self.feat_match_loss = MelganFeatureLoss()
        self.mse_loss = MSEGLoss()

    def forward(self, y_hat, y, lens, scores_fake=None, feats_fake=None, feats_real=None):
        return_dict = {}

        # Waveform loss
        return_dict["l1_wavform"] = self.l1_masked(y_hat, y, lens)
        return_dict["loss"] = return_dict["l1_wavform"] * 10

        # Spectrogram losses
        return_dict["G_stft_loss_mg"], return_dict["G_stft_loss_sc"] = self.stft_loss(y_hat.squeeze(1), y.squeeze(1))
        return_dict["loss"] += (return_dict["G_stft_loss_mg"] + return_dict["G_stft_loss_sc"]) * 1

        if scores_fake is not None and feats_fake is not None and feats_real is not None:
            # Feature matching loss
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            return_dict["feat_match"] = feat_match_loss
            return_dict["loss"] += feat_match_loss * 108

            # MSE adversarial loss
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mse_loss)
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
        self.mse_loss = MSEDLoss()

    def forward(self, scores_fake, scores_real):
        return_dict = {}
        mse_D_loss, mse_D_real_loss, mse_D_fake_loss = _apply_D_loss(
            scores_fake=scores_fake, scores_real=scores_real, loss_func=self.mse_loss
        )
        return_dict["D_mse"] = mse_D_loss
        return_dict["D_mse_real"] = mse_D_real_loss
        return_dict["D_mse_fake"] = mse_D_fake_loss
        return_dict["loss"] = return_dict["D_mse"] 
        return return_dict