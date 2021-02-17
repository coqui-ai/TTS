from torch import nn
import torch.nn.functional as F
from TTS.vocoder.models.melgan_multiscale_discriminator import MelganMultiscaleDiscriminator

class PeriodDiscriminator(nn.Module):

    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()

        self.period = period
        self.discriminator = nn.ModuleList([
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(1, 64, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(64, 128, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(128, 256, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(256, 512, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.Sequential(
                nn.utils.weight_norm(nn.Conv2d(512, 1024, kernel_size=(5, 1))),
                nn.LeakyReLU(0.2, inplace=True),
            ),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1))),
        ])


    def forward(self, x):
        batch_size = x.shape[0]
        pad = self.period - (x.shape[-1] % self.period)
        x = F.pad(x, (0, pad), "reflect")
        y = x.view(batch_size, -1, self.period).contiguous()
        y = y.unsqueeze(1)
        features = list()
        for module in self.discriminator:
            y = module(y)
            features.append(y)
        return features[-1], features[:-1]


class HiFiDiscriminator(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11]):
        super(HiFiDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([ PeriodDiscriminator(periods[0]),
                                              PeriodDiscriminator(periods[1]),
                                              PeriodDiscriminator(periods[2]),
                                              PeriodDiscriminator(periods[3]),
                                              PeriodDiscriminator(periods[4]),
                                            ])

        self.msd = MelganMultiscaleDiscriminator()

    def forward(self, x):
        scores, feats = self.msd(x)
        for key, disc in enumerate(self.discriminators):
            score, feat = disc(x)
            scores.append(score)
            feats.append(feat)
        return scores, feats
