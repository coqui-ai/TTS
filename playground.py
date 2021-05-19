#%%
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from shutil import copyfile
# %%
# meta_dir = '/home/big-boy/Data/blizzard2013/segmented/metadata.csv'
meta_dir = '/home/big-boy/Data/LJSpeech-1.1/metadata.csv'

df = pd.read_csv(Path(meta_dir), header=None, names=['ID', 'Text', 'T'], sep="|", delimiter=None)
df.drop(['T'], axis=1, inplace=True)
random_df = df.sample(n=12)
print([len(t) for t in random_df['Text']])
df.head()

# %%
for i in random_df['ID']:
    # to_copy = Path('/home/big-boy/Data/blizzard2013/segmented/wavs/' + i + '.wav')
    # destination = Path('/home/big-boy/Data/blizzard2013/segmented/refs/seen/' + i + '.wav')
    to_copy = Path('/home/big-boy/Data/LJSpeech-1.1//wavs/' + i + '.wav')
    destination = Path('/home/big-boy/Data/LJSpeech-1.1/refs/seen/' + i + '.wav')
    copyfile(to_copy, destination)

#%%
random_df['data'] = random_df[['ID', 'Text']].agg(lambda x: "|".join(map(str, x)), axis=1)
random_df['data'].to_csv(Path('/home/big-boy/Data/LJSpeech-1.1/refs_metadata.csv'), sep='\n', header=None, index=False)

# %%
# Read playground
dff = pd.read_csv(Path('/home/big-boy/Data/LJSpeech-1.1/refs_metadata.csv'), header=None, names=['ID', 'Text'], sep='|', delimiter=None)

for row in dff.iterrows():
    print(row[0])
    print(row[1]['ID'])

# %%
def extract_axis_1(data, ind):
    """
    Get specified elements along the first axis of tensor
    :param data: Tensor that will be subsetted [N, T_in, embed_dim]
    :param ind: Indices to take [N]
    :return: Subsetted tensor [N, embed_dim]
    """
    ind = torch.LongTensor(ind) - 1
    select_idx = data.index_select(1, ind) # select indices
    extracted_axis = torch.diagonal(select_idx) # diagonal contains the extracted data
    print(extracted_axis)
    return torch.transpose(extracted_axis, 0, 1) # reshape into [N, embed_dim]


# %%
print(extract_axis_1(torch.tensor(matrix), to_extract))
# %%

x = torch.randn(1, 1, 125, 4) # [batch_size, 1, time_dim, embed_dim]

S = 2
W = 1 # in channels
Filter = 8 # out channels / filter size
P = int(np.ceil(((S-1)*W-S+Filter)/2))
# x_pad = F.pad(x, (P//2, P//2, P//2, P//2))  # [left, right, top, bot]
x_pad = F.pad(x, (1, 1, 1, 1))  # [left, right, top, bot]
# print('x shape: ', x.shape)
# print('x_pad shape: ', x_pad.shape)
# print(x_pad[0, 0, :, 1])

filters = [1] + [2, 2, 4, 4, 6, 6]

valid_length = torch.tensor(100)
for i in range(len(filters)-1):
    x = torch.nn.Conv2d(in_channels=filters[i],
                        out_channels=filters[i+1],
                        kernel_size=(3, 3),
                        stride=(2, 2),
                        padding=(1, 1))(x)
    valid_length = torch.tensor([torch.ceil(valid_length/2)])
    post_conv_max_width = x.size(2)
    mask = torch.arange(post_conv_max_width).expand(1, post_conv_max_width) < valid_length.unsqueeze(1)
    mask = mask.expand(1, 1, -1, -1).transpose(2, 0).transpose(-1, 2) # [batch_size, 1, post_conv_max_width, 1]
    print('unmasked: ', x[0, 0, :, :])
    print('_____________')
    x = x*mask
    print('masked: ', x[0, 0, :, :])
    print('#############')


# padded_output_shape = torch.nn.Conv2d(in_channels=1,
#                                       out_channels=8,
#                                       kernel_size=(3, 3),
#                                       stride=(2, 2))(x_pad).shape

# print(P)

# print('input shape:           ', x.shape)
# print('padded input shape:    ', x_pad.shape)
# print('unpadded output shape: ', unpadded_output_shape)
# print('padded output shape:   ', padded_output_shape)

# %%
def calculate_post_conv_height(height, kernel_size, stride, pad,
                               n_convs):
    """Height of spec after n convolutions with fixed kernel/stride/pad."""
    for _ in range(n_convs):
        height = (height - kernel_size + 2 * pad) // stride + 1
    return height


calculate_post_conv_height(80, 3, 2, 2, 6)
# %%
x = torch.randn(2, 1, 5, 3)
conv = nn.Conv2d(
    in_channels=1,
    out_channels=8,
    kernel_size=(3, 3),
    stride=(2, 2),
    padding=(1, 1))

print('input shape: ', x.shape)
print('output shape: ', conv(x).shape)


#%%
x = torch.randn(5, 2, 14, 4)
_ones = torch.ones(5, 2, 5, 4)
_zeros = torch.zeros(5, 2, 1, 4)

mask = torch.cat((_ones, _zeros), 2)
print(x * mask)
