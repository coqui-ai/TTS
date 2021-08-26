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


def apply_gradient_allreduce(module):

    # sync model parameters
    for p in module.state_dict().values():
        if not torch.is_tensor(p):
            continue
        dist.broadcast(p, 0)

    def allreduce_params():
        if module.needs_reduction:
            module.needs_reduction = False
            # bucketing params based on value types
            buckets = {}
            for param in module.parameters():
                if param.requires_grad and param.grad is not None:
                    tp = type(param.data)
                    if tp not in buckets:
                        buckets[tp] = []
                    buckets[tp].append(param)
            for tp in buckets:
                bucket = buckets[tp]
                grads = [param.grad.data for param in bucket]
                coalesced = _flatten_dense_tensors(grads)
                dist.all_reduce(coalesced, op=dist.reduce_op.SUM)
                coalesced /= dist.get_world_size()
                for buf, synced in zip(grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):

        def allreduce_hook(*_):
            Variable._execution_engine.queue_callback(allreduce_params)  # pylint: disable=protected-access

        if param.requires_grad:
            param.register_hook(allreduce_hook)

    def set_needs_reduction(self, *_):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module
