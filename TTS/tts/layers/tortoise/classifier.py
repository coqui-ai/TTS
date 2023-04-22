import torch
import torch.nn as nn

from TTS.tts.layers.tortoise.arch_utils import AttentionBlock, Downsample, Upsample, normalization, zero_module


class ResBlock(nn.Module):
    def __init__(
        self,
        channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        up=False,
        down=False,
        kernel_size=3,
        do_checkpoint=True,
    ):
        super().__init__()
        self.channels = channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_scale_shift_norm = use_scale_shift_norm
        self.do_checkpoint = do_checkpoint
        padding = 1 if kernel_size == 3 else 2

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            nn.Conv1d(channels, self.out_channels, kernel_size, padding=padding),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(nn.Conv1d(self.out_channels, self.out_channels, kernel_size, padding=padding)),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = nn.Conv1d(dims, channels, self.out_channels, kernel_size, padding=padding)
        else:
            self.skip_connection = nn.Conv1d(dims, channels, self.out_channels, 1)

    def forward(self, x):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        h = self.out_layers(h)
        return self.skip_connection(x) + h


class AudioMiniEncoder(nn.Module):
    def __init__(
        self,
        spec_dim,
        embedding_dim,
        base_channels=128,
        depth=2,
        resnet_blocks=2,
        attn_blocks=4,
        num_attn_heads=4,
        dropout=0,
        downsample_factor=2,
        kernel_size=3,
    ):
        super().__init__()
        self.init = nn.Sequential(nn.Conv1d(spec_dim, base_channels, 3, padding=1))
        ch = base_channels
        res = []
        self.layers = depth
        for l in range(depth):
            for r in range(resnet_blocks):
                res.append(ResBlock(ch, dropout, do_checkpoint=False, kernel_size=kernel_size))
            res.append(Downsample(ch, use_conv=True, out_channels=ch * 2, factor=downsample_factor))
            ch *= 2
        self.res = nn.Sequential(*res)
        self.final = nn.Sequential(normalization(ch), nn.SiLU(), nn.Conv1d(ch, embedding_dim, 1))
        attn = []
        for a in range(attn_blocks):
            attn.append(AttentionBlock(embedding_dim, num_attn_heads, do_checkpoint=False))
        self.attn = nn.Sequential(*attn)
        self.dim = embedding_dim

    def forward(self, x):
        h = self.init(x)
        h = self.res(h)
        h = self.final(h)
        for blk in self.attn:
            h = blk(h)
        return h[:, :, 0]


class AudioMiniEncoderWithClassifierHead(nn.Module):
    def __init__(self, classes, distribute_zero_label=True, **kwargs):
        super().__init__()
        self.enc = AudioMiniEncoder(**kwargs)
        self.head = nn.Linear(self.enc.dim, classes)
        self.num_classes = classes
        self.distribute_zero_label = distribute_zero_label

    def forward(self, x, labels=None):
        h = self.enc(x)
        logits = self.head(h)
        if labels is None:
            return logits
        else:
            if self.distribute_zero_label:
                oh_labels = nn.functional.one_hot(labels, num_classes=self.num_classes)
                zeros_indices = (labels == 0).unsqueeze(-1)
                # Distribute 20% of the probability mass on all classes when zero is specified, to compensate for dataset noise.
                zero_extra_mass = torch.full_like(
                    oh_labels,
                    dtype=torch.float,
                    fill_value=0.2 / (self.num_classes - 1),
                )
                zero_extra_mass[:, 0] = -0.2
                zero_extra_mass = zero_extra_mass * zeros_indices
                oh_labels = oh_labels + zero_extra_mass
            else:
                oh_labels = labels
            loss = nn.functional.cross_entropy(logits, oh_labels)
            return loss
