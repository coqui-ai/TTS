import numpy as np
import torch
import torch.nn.functional as F

from scipy import signal as sig


# adapted from
# https://github.com/kan-bayashi/ParallelWaveGAN/tree/master/parallel_wavegan
class PQMF(torch.nn.Module):
    def __init__(self, N=4, taps=62, cutoff=0.15, beta=9.0):
        super(PQMF, self).__init__()

        self.N = N
        self.taps = taps
        self.cutoff = cutoff
        self.beta = beta

        QMF = sig.firwin(taps + 1, cutoff, window=('kaiser', beta))
        H = np.zeros((N, len(QMF)))
        G = np.zeros((N, len(QMF)))
        for k in range(N):
            constant_factor = (2 * k + 1) * (np.pi /
                                             (2 * N)) * (np.arange(taps + 1) -
                                                         ((taps - 1) / 2))  # TODO: (taps - 1) -> taps
            phase = (-1)**k * np.pi / 4
            H[k] = 2 * QMF * np.cos(constant_factor + phase)

            G[k] = 2 * QMF * np.cos(constant_factor - phase)

        H = torch.from_numpy(H[:, None, :]).float()
        G = torch.from_numpy(G[None, :, :]).float()

        self.register_buffer("H", H)
        self.register_buffer("G", G)

        updown_filter = torch.zeros((N, N, N)).float()
        for k in range(N):
            updown_filter[k, k, 0] = 1.0
        self.register_buffer("updown_filter", updown_filter)
        self.N = N

        self.pad_fn = torch.nn.ConstantPad1d(taps // 2, 0.0)

    def forward(self, x):
        return self.analysis(x)

    def analysis(self, x):
        return F.conv1d(x, self.H, padding=self.taps // 2, stride=self.N)

    def synthesis(self, x):
        x = F.conv_transpose1d(x,
                               self.updown_filter * self.N,
                               stride=self.N)
        x = F.conv1d(x, self.G, padding=self.taps // 2)
        return x
