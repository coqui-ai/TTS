# edited from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/distributed.py
import torch
import torch.distributed as dist


def reduce_tensor(tensor, num_gpus):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.reduce_op.SUM)
    rt /= num_gpus
    return rt


def init_distributed(rank, num_gpus, group_name, dist_backend, dist_url):
    assert torch.cuda.is_available(), "Distributed mode requires CUDA."

    # Set cuda device so everything is done on the right GPU.
    torch.cuda.set_device(rank % torch.cuda.device_count())

    # Initialize distributed communication
    dist.init_process_group(dist_backend, init_method=dist_url, world_size=num_gpus, rank=rank, group_name=group_name)
