import sys
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import time

# fix this
from TTS.utils.audio import AudioProcessor as ap
from TTS.vocoder.utils.distribution import (
    sample_from_gaussian,
    sample_from_discretized_mix_logistic,
)


def stream(string, variables):
    sys.stdout.write(f"\r{string}" % variables)

# pylint: disable=abstract-method
# relates https://github.com/pytorch/pytorch/issues/42305
class ResBlock(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.conv1 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(dims, dims, kernel_size=1, bias=False)
        self.batch_norm1 = nn.BatchNorm1d(dims)
        self.batch_norm2 = nn.BatchNorm1d(dims)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.batch_norm2(x)
        return x + residual


class MelResNet(nn.Module):
    def __init__(self, num_res_blocks, in_dims, compute_dims, res_out_dims, pad):
        super().__init__()
        k_size = pad * 2 + 1
        self.conv_in = nn.Conv1d(
            in_dims, compute_dims, kernel_size=k_size, bias=False)
        self.batch_norm = nn.BatchNorm1d(compute_dims)
        self.layers = nn.ModuleList()
        for _ in range(num_res_blocks):
            self.layers.append(ResBlock(compute_dims))
        self.conv_out = nn.Conv1d(compute_dims, res_out_dims, kernel_size=1)

    def forward(self, x):
        x = self.conv_in(x)
        x = self.batch_norm(x)
        x = F.relu(x)
        for f in self.layers:
            x = f(x)
        x = self.conv_out(x)
        return x


class Stretch2d(nn.Module):
    def __init__(self, x_scale, y_scale):
        super().__init__()
        self.x_scale = x_scale
        self.y_scale = y_scale

    def forward(self, x):
        b, c, h, w = x.size()
        x = x.unsqueeze(-1).unsqueeze(3)
        x = x.repeat(1, 1, 1, self.y_scale, 1, self.x_scale)
        return x.view(b, c, h * self.y_scale, w * self.x_scale)


class UpsampleNetwork(nn.Module):
    def __init__(
            self,
            feat_dims,
            upsample_scales,
            compute_dims,
            num_res_blocks,
            res_out_dims,
            pad,
            use_aux_net,
        ):
        super().__init__()
        self.total_scale = np.cumproduct(upsample_scales)[-1]
        self.indent = pad * self.total_scale
        self.use_aux_net = use_aux_net
        if use_aux_net:
            self.resnet = MelResNet(
                num_res_blocks, feat_dims, compute_dims, res_out_dims, pad
            )
            self.resnet_stretch = Stretch2d(self.total_scale, 1)
        self.up_layers = nn.ModuleList()
        for scale in upsample_scales:
            k_size = (1, scale * 2 + 1)
            padding = (0, scale)
            stretch = Stretch2d(scale, 1)
            conv = nn.Conv2d(1, 1, kernel_size=k_size,
                             padding=padding, bias=False)
            conv.weight.data.fill_(1.0 / k_size[1])
            self.up_layers.append(stretch)
            self.up_layers.append(conv)

    def forward(self, m):
        if self.use_aux_net:
            aux = self.resnet(m).unsqueeze(1)
            aux = self.resnet_stretch(aux)
            aux = aux.squeeze(1)
            aux = aux.transpose(1, 2)
        else:
            aux = None
        m = m.unsqueeze(1)
        for f in self.up_layers:
            m = f(m)
        m = m.squeeze(1)[:, :, self.indent: -self.indent]
        return m.transpose(1, 2), aux


class Upsample(nn.Module):
    def __init__(self, scale, pad, num_res_blocks, feat_dims, compute_dims,
                 res_out_dims, use_aux_net):
        super().__init__()
        self.scale = scale
        self.pad = pad
        self.indent = pad * scale
        self.use_aux_net = use_aux_net
        self.resnet = MelResNet(num_res_blocks, feat_dims,
                                compute_dims, res_out_dims, pad)

    def forward(self, m):
        if self.use_aux_net:
            aux = self.resnet(m)
            aux = torch.nn.functional.interpolate(
                aux, scale_factor=self.scale, mode="linear", align_corners=True
            )
            aux = aux.transpose(1, 2)
        else:
            aux = None
        m = torch.nn.functional.interpolate(
            m, scale_factor=self.scale, mode="linear", align_corners=True
        )
        m = m[:, :, self.indent: -self.indent]
        m = m * 0.045  # empirically found

        return m.transpose(1, 2), aux


class WaveRNN(nn.Module):
    def __init__(self,
                 rnn_dims,
                 fc_dims,
                 mode,
                 mulaw,
                 pad,
                 use_aux_net,
                 use_upsample_net,
                 upsample_factors,
                 feat_dims,
                 compute_dims,
                 res_out_dims,
                 num_res_blocks,
                 hop_length,
                 sample_rate,
                 ):
        super().__init__()
        self.mode = mode
        self.mulaw = mulaw
        self.pad = pad
        self.use_upsample_net = use_upsample_net
        self.use_aux_net = use_aux_net
        if isinstance(self.mode, int):
            self.n_classes = 2 ** self.mode
        elif self.mode == "mold":
            self.n_classes = 3 * 10
        elif self.mode == "gauss":
            self.n_classes = 2
        else:
            raise RuntimeError("Unknown model mode value - ", self.mode)

        self.rnn_dims = rnn_dims
        self.aux_dims = res_out_dims // 4
        self.hop_length = hop_length
        self.sample_rate = sample_rate

        if self.use_upsample_net:
            assert (
                np.cumproduct(upsample_factors)[-1] == self.hop_length
            ), " [!] upsample scales needs to be equal to hop_length"
            self.upsample = UpsampleNetwork(
                feat_dims,
                upsample_factors,
                compute_dims,
                num_res_blocks,
                res_out_dims,
                pad,
                use_aux_net,
            )
        else:
            self.upsample = Upsample(
                hop_length,
                pad,
                num_res_blocks,
                feat_dims,
                compute_dims,
                res_out_dims,
                use_aux_net,
            )
        if self.use_aux_net:
            self.I = nn.Linear(feat_dims + self.aux_dims + 1, rnn_dims)
            self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
            self.rnn2 = nn.GRU(rnn_dims + self.aux_dims,
                               rnn_dims, batch_first=True)
            self.fc1 = nn.Linear(rnn_dims + self.aux_dims, fc_dims)
            self.fc2 = nn.Linear(fc_dims + self.aux_dims, fc_dims)
            self.fc3 = nn.Linear(fc_dims, self.n_classes)
        else:
            self.I = nn.Linear(feat_dims + 1, rnn_dims)
            self.rnn1 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
            self.rnn2 = nn.GRU(rnn_dims, rnn_dims, batch_first=True)
            self.fc1 = nn.Linear(rnn_dims, fc_dims)
            self.fc2 = nn.Linear(fc_dims, fc_dims)
            self.fc3 = nn.Linear(fc_dims, self.n_classes)

    def forward(self, x, mels):
        bsize = x.size(0)
        h1 = torch.zeros(1, bsize, self.rnn_dims).to(x.device)
        h2 = torch.zeros(1, bsize, self.rnn_dims).to(x.device)
        mels, aux = self.upsample(mels)

        if self.use_aux_net:
            aux_idx = [self.aux_dims * i for i in range(5)]
            a1 = aux[:, :, aux_idx[0]: aux_idx[1]]
            a2 = aux[:, :, aux_idx[1]: aux_idx[2]]
            a3 = aux[:, :, aux_idx[2]: aux_idx[3]]
            a4 = aux[:, :, aux_idx[3]: aux_idx[4]]

        x = (
            torch.cat([x.unsqueeze(-1), mels, a1], dim=2)
            if self.use_aux_net
            else torch.cat([x.unsqueeze(-1), mels], dim=2)
        )
        x = self.I(x)
        res = x
        self.rnn1.flatten_parameters()
        x, _ = self.rnn1(x, h1)

        x = x + res
        res = x
        x = torch.cat([x, a2], dim=2) if self.use_aux_net else x
        self.rnn2.flatten_parameters()
        x, _ = self.rnn2(x, h2)

        x = x + res
        x = torch.cat([x, a3], dim=2) if self.use_aux_net else x
        x = F.relu(self.fc1(x))

        x = torch.cat([x, a4], dim=2) if self.use_aux_net else x
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def inference(self, mels, batched=None, target=None, overlap=None):

        self.eval()
        device = mels.device
        output = []
        start = time.time()
        rnn1 = self.get_gru_cell(self.rnn1)
        rnn2 = self.get_gru_cell(self.rnn2)

        with torch.no_grad():
            if isinstance(mels, np.ndarray):
                mels = torch.FloatTensor(mels).to(device)

            if mels.ndim == 2:
                mels = mels.unsqueeze(0)
            wave_len = (mels.size(-1) - 1) * self.hop_length

            mels = self.pad_tensor(mels.transpose(
                1, 2), pad=self.pad, side="both")
            mels, aux = self.upsample(mels.transpose(1, 2))

            if batched:
                mels = self.fold_with_overlap(mels, target, overlap)
                if aux is not None:
                    aux = self.fold_with_overlap(aux, target, overlap)

            b_size, seq_len, _ = mels.size()

            h1 = torch.zeros(b_size, self.rnn_dims).to(device)
            h2 = torch.zeros(b_size, self.rnn_dims).to(device)
            x = torch.zeros(b_size, 1).to(device)

            if self.use_aux_net:
                d = self.aux_dims
                aux_split = [aux[:, :, d * i: d * (i + 1)] for i in range(4)]

            for i in range(seq_len):

                m_t = mels[:, i, :]

                if self.use_aux_net:
                    a1_t, a2_t, a3_t, a4_t = (a[:, i, :] for a in aux_split)

                x = (
                    torch.cat([x, m_t, a1_t], dim=1)
                    if self.use_aux_net
                    else torch.cat([x, m_t], dim=1)
                )
                x = self.I(x)
                h1 = rnn1(x, h1)

                x = x + h1
                inp = torch.cat([x, a2_t], dim=1) if self.use_aux_net else x
                h2 = rnn2(inp, h2)

                x = x + h2
                x = torch.cat([x, a3_t], dim=1) if self.use_aux_net else x
                x = F.relu(self.fc1(x))

                x = torch.cat([x, a4_t], dim=1) if self.use_aux_net else x
                x = F.relu(self.fc2(x))

                logits = self.fc3(x)

                if self.mode == "mold":
                    sample = sample_from_discretized_mix_logistic(
                        logits.unsqueeze(0).transpose(1, 2)
                    )
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1).to(device)
                elif self.mode == "gauss":
                    sample = sample_from_gaussian(
                        logits.unsqueeze(0).transpose(1, 2))
                    output.append(sample.view(-1))
                    x = sample.transpose(0, 1).to(device)
                elif isinstance(self.mode, int):
                    posterior = F.softmax(logits, dim=1)
                    distrib = torch.distributions.Categorical(posterior)

                    sample = 2 * distrib.sample().float() / (self.n_classes - 1.0) - 1.0
                    output.append(sample)
                    x = sample.unsqueeze(-1)
                else:
                    raise RuntimeError(
                        "Unknown model mode value - ", self.mode)

                if i % 100 == 0:
                    self.gen_display(i, seq_len, b_size, start)

        output = torch.stack(output).transpose(0, 1)
        output = output.cpu()
        if batched:
            output = output.numpy()
            output = output.astype(np.float64)

            output = self.xfade_and_unfold(output, target, overlap)
        else:
            output = output[0]

        if self.mulaw and isinstance(self.mode, int):
            output = ap.mulaw_decode(output, self.mode)

        # Fade-out at the end to avoid signal cutting out suddenly
        fade_out = np.linspace(1, 0, 20 * self.hop_length)
        output = output[:wave_len]

        if wave_len > len(fade_out):
            output[-20 * self.hop_length:] *= fade_out

        self.train()
        return output

    def gen_display(self, i, seq_len, b_size, start):
        gen_rate = (i + 1) / (time.time() - start) * b_size / 1000
        realtime_ratio = gen_rate * 1000 / self.sample_rate
        stream(
            "%i/%i -- batch_size: %i -- gen_rate: %.1f kHz -- x_realtime: %.1f  ",
            (i * b_size, seq_len * b_size, b_size, gen_rate, realtime_ratio),
        )

    def fold_with_overlap(self, x, target, overlap):
        """Fold the tensor with overlap for quick batched inference.
            Overlap will be used for crossfading in xfade_and_unfold()
        Args:
            x (tensor)    : Upsampled conditioning features.
                            shape=(1, timesteps, features)
            target (int)  : Target timesteps for each index of batch
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (tensor) : shape=(num_folds, target + 2 * overlap, features)
        Details:
            x = [[h1, h2, ... hn]]
            Where each h is a vector of conditioning features
            Eg: target=2, overlap=1 with x.size(1)=10
            folded = [[h1, h2, h3, h4],
                      [h4, h5, h6, h7],
                      [h7, h8, h9, h10]]
        """

        _, total_len, features = x.size()

        # Calculate variables needed
        num_folds = (total_len - overlap) // (target + overlap)
        extended_len = num_folds * (overlap + target) + overlap
        remaining = total_len - extended_len

        # Pad if some time steps poking out
        if remaining != 0:
            num_folds += 1
            padding = target + 2 * overlap - remaining
            x = self.pad_tensor(x, padding, side="after")

        folded = torch.zeros(num_folds, target + 2 *
                             overlap, features).to(x.device)

        # Get the values for the folded tensor
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            folded[i] = x[:, start:end, :]

        return folded

    @staticmethod
    def get_gru_cell(gru):
        gru_cell = nn.GRUCell(gru.input_size, gru.hidden_size)
        gru_cell.weight_hh.data = gru.weight_hh_l0.data
        gru_cell.weight_ih.data = gru.weight_ih_l0.data
        gru_cell.bias_hh.data = gru.bias_hh_l0.data
        gru_cell.bias_ih.data = gru.bias_ih_l0.data
        return gru_cell

    @staticmethod
    def pad_tensor(x, pad, side="both"):
        # NB - this is just a quick method i need right now
        # i.e., it won't generalise to other shapes/dims
        b, t, c = x.size()
        total = t + 2 * pad if side == "both" else t + pad
        padded = torch.zeros(b, total, c).to(x.device)
        if side in ("before", "both"):
            padded[:, pad: pad + t, :] = x
        elif side == "after":
            padded[:, :t, :] = x
        return padded

    @staticmethod
    def xfade_and_unfold(y, target, overlap):
        """Applies a crossfade and unfolds into a 1d array.
        Args:
            y (ndarry)    : Batched sequences of audio samples
                            shape=(num_folds, target + 2 * overlap)
                            dtype=np.float64
            overlap (int) : Timesteps for both xfade and rnn warmup
        Return:
            (ndarry) : audio samples in a 1d array
                       shape=(total_len)
                       dtype=np.float64
        Details:
            y = [[seq1],
                 [seq2],
                 [seq3]]
            Apply a gain envelope at both ends of the sequences
            y = [[seq1_in, seq1_target, seq1_out],
                 [seq2_in, seq2_target, seq2_out],
                 [seq3_in, seq3_target, seq3_out]]
            Stagger and add up the groups of samples:
            [seq1_in, seq1_target, (seq1_out + seq2_in), seq2_target, ...]
        """

        num_folds, length = y.shape
        target = length - 2 * overlap
        total_len = num_folds * (target + overlap) + overlap

        # Need some silence for the rnn warmup
        silence_len = overlap // 2
        fade_len = overlap - silence_len
        silence = np.zeros((silence_len), dtype=np.float64)

        # Equal power crossfade
        t = np.linspace(-1, 1, fade_len, dtype=np.float64)
        fade_in = np.sqrt(0.5 * (1 + t))
        fade_out = np.sqrt(0.5 * (1 - t))

        # Concat the silence to the fades
        fade_in = np.concatenate([silence, fade_in])
        fade_out = np.concatenate([fade_out, silence])

        # Apply the gain to the overlap samples
        y[:, :overlap] *= fade_in
        y[:, -overlap:] *= fade_out

        unfolded = np.zeros((total_len), dtype=np.float64)

        # Loop to add up all the samples
        for i in range(num_folds):
            start = i * (target + overlap)
            end = start + target + 2 * overlap
            unfolded[start:end] += y[i]

        return unfolded

    def load_checkpoint(self, config, checkpoint_path, eval=False):  # pylint: disable=unused-argument, redefined-builtin
        state = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.load_state_dict(state['model'])
        if eval:
            self.eval()
            assert not self.training
