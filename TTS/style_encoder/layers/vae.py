import torch
import torch.nn.functional as F
from torch import nn

class VAEStyleEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.
    See: https://arxiv.org/pdf/1812.04342.pdf
    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim, latent_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim, batch_first=True
        )

        self.mu = nn.Linear(embedding_dim, latent_dim)
        self.logvar = nn.Linear(embedding_dim, latent_dim)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        mu = self.mu(out.squeeze(0))
        logvar = self.logvar(out.squeeze(0))

        z = self.reparametrize(mu, logvar)

        z = z.unsqueeze(1)
        
        return {'z' :z , 'mean': mu, 'log_var': logvar, 'out': out} #out will be the RE output, just to have the output if needed

    def reparametrize(self, mu, logvar):
        std = torch.exp(0.5*logvar)  
        eps = torch.randn_like(std)

        return eps.mul(std).add_(mu)


    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height


## Not well implemented yet
class ModifiedReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram, modified to use a last linear layer and bilstm.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, embedding_dim):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(1, 1)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 1, num_layers)
        self.recurrence = nn.GRU(
            input_size=filters[-1] * post_conv_height, hidden_size=embedding_dim , batch_first=True, bidirectional=True
        )

        self.dense = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, inputs):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)
        # x: 4D tensor [batch_size, num_channels==1, num_frames, num_mel]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]
        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]
        self.recurrence.flatten_parameters()
        _, out = self.recurrence(x)
        # out: 3D tensor [seq_len==1, batch_size, encoding_size=128]

        out = self.dense(out.squeeze(0))

        return out

    @staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad, n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height