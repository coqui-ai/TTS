#%%
import torch
import numpy as np
# %%
matrix = np.array([[1, 2, 3, 4, 4, 5], [5, 2, 6, 7, 83, 6], [1, 5, 3, 0, 0, 0]])
to_extract = np.array([3, 6, 2])
print(matrix)
print(to_extract)
print('correct: ', [3, 6, 5])

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
