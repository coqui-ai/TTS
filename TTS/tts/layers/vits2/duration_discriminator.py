import torch
from torch import nn
from TTS.tts.layers.generic.normalization import LayerNorm2


class DurationDiscriminator(nn.Module): #vits2
    """VITS-2 Duration Discriminator.
    
    ::
      dur_r, dur_hat -> DurationDiscriminator() -> output_probs  

    Args:
        in_channels (int): number of input channels.
        filter_channels (int): number of filter channels.
        kernel_size (int): kernel size of the convolutional layers.
        p_dropout (float): dropout probability.
        gin_channels (int): number of global conditioning channels. 
                            Unused for now. 
    
    Returns:
        List[Tensor]: list of discriminator scores. Real, Predicted/Generated.
    """
  # TODO : not using "spk conditioning" for now according to the paper.
  # Can be a better discriminator if we use it.
    def __init__(
            self, 
            in_channels, 
            filter_channels, 
            kernel_size, 
            p_dropout, 
            gin_channels=0
            ):
        super().__init__()

        self.in_channels = in_channels
        self.filter_channels = filter_channels
        self.kernel_size = kernel_size
        self.p_dropout = p_dropout
        self.gin_channels = gin_channels

        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.dur_proj = nn.Conv1d(1, filter_channels, 1)

        self.pre_out_conv_1 = nn.Conv1d(2*filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.pre_out_norm_1 = LayerNorm2(filter_channels)
        self.pre_out_conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size//2)
        self.pre_out_norm_2 = LayerNorm2(filter_channels)

        # if gin_channels != 0:
        #   self.cond = nn.Conv1d(gin_channels, in_channels, 1)

        self.output_layer = nn.Sequential(
            nn.Linear(filter_channels, 1), 
            nn.Sigmoid() 
        )

    def forward_probability(self, x, x_mask, dur, g=None):
        dur = self.dur_proj(dur)
        x = torch.cat([x, dur], dim=1)
        x = self.pre_out_conv_1(x * x_mask)
        x = self.pre_out_conv_2(x * x_mask)
        x = x * x_mask
        x = x.transpose(1, 2)
        output_prob = self.output_layer(x)
        return output_prob

    def forward(self, x, x_mask, dur_r, dur_hat, g=None):
        x = torch.detach(x)
        # if g is not None:
        #   g = torch.detach(g)
        #   x = x + self.cond(g)
        x = self.conv_1(x * x_mask)
        # x = self.drop(x)
        x = self.conv_2(x * x_mask)
        # x = self.drop(x)

        output_probs = []
        for dur in [dur_r, dur_hat]:
            output_prob = self.forward_probability(x, x_mask, dur, g)
            output_probs.append(output_prob)

        return output_probs