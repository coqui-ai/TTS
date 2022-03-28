from typing import Type
import torch
from TTS.utils.audio import TorchSTFT
from TTS.tts.layers.losses import L1LossMasked

class BWELoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.l1_wavform = L1LossMasked(False)
        self.l1_mel = L1LossMasked(False)
        self.l1_spec = [L1LossMasked(False) for _ in range(4)]
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

    def forward(self, y_hat, y, lens):
        return_dict = {}
        return_dict["l1_wavform"] = self.l1_wavform(y_hat, y, lens)
        with torch.no_grad():
            y_specs = [self.specs[i](y[:, 0, :]) for i in range(4)]
            y_mel = self.mel_spec(y[:, 0, :])
        y_hat_specs = [self.specs[i](y_hat[:, 0, :]) for i in range(4)]
        y_hat_mel = self.mel_spec(y_hat[:, 0, :])
        mel_lens = self.compute_lens(y_mel, lens)
        return_dict["l1_mel"] = self.l1_mel(y_hat_mel, y_mel, mel_lens)
        return_dict["l1_spec"] = torch.mean(torch.stack([
            self.l1_spec[i](y_hat_specs[i], y_specs[i], self.compute_lens(y_specs[i], lens))
            for i in range(4)]))
        return_dict["loss"] = return_dict["l1_wavform"] * 10 + return_dict["l1_mel"] + return_dict["l1_spec"]

        # check if any loss is NaN
        for key, loss in return_dict.items():
            if torch.isnan(loss):
                raise RuntimeError(f" [!] NaN loss with {key}.")
        return return_dict

    def compute_lens(self, spec, wav_lens):
        BS = wav_lens.shape[0]
        max_len = torch.max(wav_lens)
        return torch.stack([ spec.shape[-1] * wav_lens[i] / max_len  for i in range(BS)])
