import torch
from torch import nn
from torch.nn import functional as F

class ReferenceEncoder(nn.Module):
    """NN module creating a fixed size prosody embedding from a spectrogram.

    inputs: mel spectrograms [batch_size, num_spec_frames, num_mel]
    outputs: [batch_size, embedding_dim]
    """

    def __init__(self, num_mel, filter):
        super().__init__()
        self.num_mel = num_mel
        start_index = 2 
        end_index = filter / 16
        i = start_index
        filt_len = []
        while i <= end_index:
            i = i * 2
            filt_len.append(i)
            filt_len.append(i)
        filters = [1] + filt_len
        num_layers = len(filters) - 1
        convs = [
            nn.Conv2d(
                in_channels=filters[i], out_channels=filters[i + 1], kernel_size=(3, 3), stride=(2, 2), padding=(2, 2)
            )
            for i in range(num_layers)
        ]
        self.convs = nn.ModuleList(convs)
        self.training = False
        self.bns = nn.ModuleList([nn.BatchNorm2d(num_features=filter_size) for filter_size in filters[1:]])

        post_conv_height = self.calculate_post_conv_height(num_mel, 3, 2, 2, num_layers)
        self.recurrence = nn.LSTM(
            input_size=filters[-1] * post_conv_height, hidden_size=out_dim, batch_first=True, bidirectional=False
        )

    def forward(self, inputs, input_lengths):
        batch_size = inputs.size(0)
        x = inputs.view(batch_size, 1, -1, self.num_mel)  # [batch_size, num_channels==1, num_frames, num_mel]
        valid_lengths = input_lengths.float()  # [batch_size]
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
            valid_lengths = (valid_lengths / 2).float()
            valid_lengths = torch.ceil(valid_lengths).to(dtype=torch.int64) + 1  # 2 is stride -- size: [batch_size]
            post_conv_max_width = x.size(2)

            mask = torch.arange(post_conv_max_width).to(inputs.device).expand(
                len(valid_lengths), post_conv_max_width
            ) < valid_lengths.unsqueeze(1)
            mask = mask.expand(1, 1, -1, -1).transpose(2, 0).transpose(-1, 2)  # [batch_size, 1, post_conv_max_width, 1]
            x = x * mask

        x = x.transpose(1, 2)
        # x: 4D tensor [batch_size, post_conv_width,
        #               num_channels==128, post_conv_height]

        post_conv_width = x.size(1)
        x = x.contiguous().view(batch_size, post_conv_width, -1)
        # x: 3D tensor [batch_size, post_conv_width,
        #               num_channels*post_conv_height]

        # Routine for fetching the last valid output of a dynamic LSTM with varying input lengths and padding
        post_conv_input_lengths = valid_lengths
        packed_seqs = nn.utils.rnn.pack_padded_sequence(
            x, post_conv_input_lengths.tolist(), batch_first=True, enforce_sorted=False
        )  # dynamic rnn sequence padding
        self.recurrence.flatten_parameters()
        _, (ht, _) = self.recurrence(packed_seqs)
        last_output = ht[-1]

        return last_output.to(inputs.device)  # [B, 128]