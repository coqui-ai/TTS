from typing import Type
import torch
from TTS.utils.audio import TorchSTFT
from TTS.tts.layers.losses import L1LossMasked
from TTS.vocoder.layers.losses import MelganFeatureLoss, MSEGLoss, MSEDLoss, _apply_G_adv_loss, _apply_D_loss

class BWEGeneratorLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_masked= L1LossMasked(False)
        self.specs = [TorchSTFT(
                        n_fft=512*2**i, 
                        hop_length=int(512*2**i/4),
                        win_length=512*2**i,
                        sample_rate=48000,
                        device=device,
                        do_amp_to_db=True) for i in range(4)]
        self.mel_spec = TorchSTFT(
                            n_fft=2048, 
                            hop_length=512, 
                            win_length=2048, 
                            sample_rate=48000, 
                            n_mels=128,
                            device=device,
                            do_amp_to_db=True)
        self.feat_match_loss = MelganFeatureLoss()
        self.mse_loss = MSEGLoss()

    def forward(self, y_hat, y, lens, scores_fake=None, feats_fake=None, feats_real=None):
        return_dict = {}

        # Waveform loss
        return_dict["l1_wavform"] = self.l1_masked(y_hat, y, lens)
        return_dict["loss"] = return_dict["l1_wavform"] * 50

        # Compute spectrograms
        y_specs = [self.specs[i](y[:, 0, :]) for i in range(4)]
        y_mel = self.mel_spec(y[:, 0, :])

        y_hat_specs = [self.specs[i](y_hat[:, 0, :]) for i in range(4)]
        y_hat_mel = self.mel_spec(y_hat[:, 0, :])

        # Spectrogram loss
        mel_lens = self.compute_lens(y_mel, lens)
        return_dict["l1_mel"] = self.l1_masked(y_hat_mel, y_mel, mel_lens)
        return_dict["loss"] = return_dict["l1_mel"]

        return_dict["l1_spec"] = torch.mean(torch.stack([
            self.l1_masked(y_hat_specs[i], y_specs[i], self.compute_lens(y_specs[i], lens))
            for i in range(4)]))
        return_dict["loss"] += return_dict["l1_spec"]

        if scores_fake is not None and feats_fake is not None and feats_real is not None:
            # Feature matching loss
            feat_match_loss = self.feat_match_loss(feats_fake, feats_real)
            return_dict["feat_match"] = feat_match_loss
            return_dict["loss"] += feat_match_loss * 108

            # MSE adversarial loss
            mse_fake_loss = _apply_G_adv_loss(scores_fake, self.mse_loss)
            return_dict["G_mse_fake"] = mse_fake_loss
            return_dict["loss"] += return_dict["G_mse_fake"] * 2.5

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict

    def compute_lens(self, spec, wav_lens):
        BS = wav_lens.shape[0]
        max_len = torch.max(wav_lens)
        return torch.stack([ spec.shape[-1] * wav_lens[i] / max_len  for i in range(BS)])


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
