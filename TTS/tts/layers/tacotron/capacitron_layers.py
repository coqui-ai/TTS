import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal as MVN

class CapacitronVAE(nn.Module):
    """Effective Use of Variational Embedding Capacity for prosody transfer.

    See https://arxiv.org/abs/1906.03402 """

    def __init__(self, num_mel, capacitron_embedding_dim, speaker_embedding_dim=None, text_summary_embedding_dim=None):
        super().__init__()
        # Init distributions
        self.prior_distribution = MVN(torch.zeros(capacitron_embedding_dim).to(device=torch.device('cuda')), torch.eye(capacitron_embedding_dim).to(device=torch.device('cuda')))
        self.approximate_posterior_distribution = None
        self.encoder = ReferenceEncoder(num_mel)

        # Init beta, the lagrange-like term for the KL distribution
        self.beta = torch.nn.Parameter(torch.log(torch.exp(torch.Tensor([1.0])) - 1), requires_grad=True)

        mlp_input_dimension = 128 # output size of the encoder

        if text_summary_embedding_dim is not None:
            self.text_summary_net = TextSummary(text_summary_embedding_dim)
            mlp_input_dimension += text_summary_embedding_dim
        if speaker_embedding_dim is not None:
            # TODO: Figure out what to do with speaker_embedding_dim
            mlp_input_dimension += speaker_embedding_dim

        self.post_encoder_mlp = PostEncoderMLP(mlp_input_dimension, capacitron_embedding_dim)

    def forward(self, reference_mel_info=None, text_info=None, speaker_embedding=None):
        # Use reference
        if reference_mel_info is not None:
            reference_mels = reference_mel_info[0] # [batch_size, num_frames, num_mels]
            mel_lengths = reference_mel_info[1] # [batch_size]
            enc_out = self.encoder(reference_mels, mel_lengths)

            # concat speaker_embedding and/or text summary embedding
            if text_info is not None:
                text_inputs = text_info[0] # [batch_size, num_characters, num_embedding]
                input_lengths = text_info[1]
                text_summary_out = self.text_summary_net(text_inputs, input_lengths).cuda()
                enc_out = torch.cat([enc_out, text_summary_out], dim=-1)
            if speaker_embedding is not None:
                enc_out = torch.cat([enc_out, speaker_embedding], dim=-1)

            # Feed the output of the ref encoder and information about text/speaker into
            # an MLP to produce the parameteres for the approximate poterior distributions
            mu, sigma = self.post_encoder_mlp(enc_out)
            mu.to(device=torch.device('cuda'))
            sigma.to(device=torch.device('cuda'))

            # Sample from the posterior: z ~ q(z|x)
            self.approximate_posterior_distribution = MVN(mu, torch.diag_embed(sigma))
            VAE_embedding = self.approximate_posterior_distribution.rsample()
        # Infer from the model, bypasses encoding
        else:
            # Sample from the prior: z ~ p(z)
            VAE_embedding = self.prior_distribution.sample().unsqueeze(0)

        # reshape to [batch_size, 1, capacitron_embedding_dim]
        return VAE_embedding.unsqueeze(1), self.approximate_posterior_distribution, self.prior_distribution, self.beta


class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel):

        super().__init__()
        self.num_mel = num_mel
        filters = [1] + [32, 32, 64, 64, 128, 128]
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i],
                out_channels=filters[i + 1],
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(2, 2)) for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.training = False
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_features=filter_size)
            for filter_size in filters[1:]
        ])

        post_conv_height = self.calculate_post_conv_height(
            num_mel, 3, 2, 2, num_layers)
        self.recurrence = nn.LSTM(
            input_size=filters[-1] * post_conv_height,
            hidden_size=128,
            batch_first=True,
            bidirectional=False)

    def forward(self, inputs, input_lengths):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel) # [batch_size, num_channels==1, num_frames, num_mel]
        valid_lengths = input_lengths.cpu() # [batch_size]
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x)
            x = bn(x)
            x = F.relu(x)

            # Create the post conv width mask based on the valid lengths of the output of the convolution.
            # The valid lengths for the output of a convolution on varying length inputs is
            # ceil(input_length/stride) + 1 for stride=3 and padding=2
            # For example (kernel_size=3, stride=2, padding=2):
            # 0 0 x x x x x 0 0 -> Input = 5, 0 is zero padding, x is valid values coming from padding=2 in conv2d
            # _____
            #   x _____
            #       x _____
            #           x  ____
            #               x
            # x x x x -> Output valid length = 4
            # Since every example in te batch is zero padded and therefore have separate valid_lengths,
            # we need to mask off all the values AFTER the valid length for each example in the batch.
            # Otherwise, the convolutions create noise and a lot of not real information

            valid_lengths = torch.ceil(valid_lengths/2).to(dtype=torch.int64) + 1 # 2 is stride -- size: [batch_size]
            post_conv_max_width = x.size(2)

            mask = torch.arange(post_conv_max_width).expand(len(valid_lengths), post_conv_max_width) < valid_lengths.unsqueeze(1)
            mask = mask.expand(1, 1, -1, -1).transpose(2, 0).transpose(-1, 2) # [batch_size, 1, post_conv_max_width, 1]
            x = x*mask.cuda()

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]

        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        post_conv_input_lengths = valid_lengths
        packed_seqs = nn.utils.rnn.pack_padded_sequence(x, post_conv_input_lengths.tolist(), batch_first=True, enforce_sorted=False) # dynamic rnn sequence padding
        self.recurrence.flatten_parameters()
        _, (ht, _) = self.recurrence(packed_seqs)
        last_output = ht[-1]

        return last_output.cuda() # [B, 128]

    @ staticmethod
    def calculate_post_conv_height(height, kernel_size, stride, pad,
                                   n_convs):
        """Height of spec after n convolutions with fixed kernel/stride/pad."""
        for _ in range(n_convs):
            height = (height - kernel_size + 2 * pad) // stride + 1
        return height

class TextSummary(nn.Module):
    def __init__(self, embedding_dim):
        super(TextSummary, self).__init__()
        self.lstm = nn.LSTM(256, # 256 is the hardcoded text embedding dimension from the text encoder
                            embedding_dim, # fixed length output summary the lstm creates from the input
                            batch_first=True,
                            bidirectional=False)

    def forward(self, inputs, input_lengths):
        # TODO deal with inference - input lengths is not a tensor but just a single number
        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        packed_seqs = nn.utils.rnn.pack_padded_sequence(inputs, input_lengths.tolist(), batch_first=True, enforce_sorted=False) # dynamic rnn sequence padding
        self.lstm.flatten_parameters()
        _, (ht, _) = self.lstm(packed_seqs)
        last_output = ht[-1]
        return last_output

class PostEncoderMLP(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PostEncoderMLP, self).__init__()
        self.hidden_size = hidden_size
        modules = [
            nn.Linear(input_size, hidden_size), # Hidden Layer
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size * 2)] # Output layer twice the size for mean and variance
        self.net = nn.Sequential(*modules)
        self.softplus = nn.Softplus()

    def forward(self, _input):
        mlp_output = self.net(_input)
        # The mean parameter is unconstrained
        mu = mlp_output[:, :self.hidden_size]
        # The standard deviation must be positive. Parameterise with a softplus
        sigma = self.softplus(mlp_output[:, self.hidden_size:])
        return mu, sigma
