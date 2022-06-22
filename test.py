import torch
x = torch.tensor([1,2], dtype=torch.long, device='cuda:0')
print(x[0])