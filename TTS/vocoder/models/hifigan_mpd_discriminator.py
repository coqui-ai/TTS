from torch import nn
import torch.nn.functional as F


class PeriodDiscriminator(nn.Module):

    def __init__(self, period):
        super(PeriodDiscriminator, self).__init__()
        layer = []
        self.period = period
        inp = 1
        for l in range(4):
            out = int(2 ** (5 + l + 1))
            layer += [
                nn.utils.weight_norm(nn.Conv2d(inp, out, kernel_size=(5, 1), stride=(3, 1))),
                nn.LeakyReLU(0.2)
            ]
            inp = out
        self.layer = nn.Sequential(*layer)
        self.output = nn.Sequential(
            nn.utils.weight_norm(nn.Conv2d(out, 1024, kernel_size=(5, 1))),
            nn.LeakyReLU(0.2),
            nn.utils.weight_norm(nn.Conv2d(1024, 1, kernel_size=(3, 1)))
        )

    def forward(self, x):
        batch_size = x.shape[0]
        pad = self.period - (x.shape[-1] % self.period)
        x = F.pad(x, (0, pad), "reflect")
        y = x.view(batch_size, -1, self.period).contiguous()
        y = y.unsqueeze(1)
        out1 = self.layer(y)
        return self.output(out1)


class MPD(nn.Module):
    def __init__(self, periods=[2, 3, 5, 7, 11], segment_length=16000):
        super(MPD, self).__init__()
        self.mpd1 = PeriodDiscriminator(periods[0])
        self.mpd2 = PeriodDiscriminator(periods[1])
        self.mpd3 = PeriodDiscriminator(periods[2])
        self.mpd4 = PeriodDiscriminator(periods[3])
        self.mpd5 = PeriodDiscriminator(periods[4])

    def forward(self, x):
        out1 = self.mpd1(x)
        out2 = self.mpd2(x)
        out3 = self.mpd3(x)
        out4 = self.mpd4(x)
        out5 = self.mpd5(x)
        return out1, out2, out3, out4, out5
