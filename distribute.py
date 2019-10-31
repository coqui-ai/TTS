# edited from https://github.com/fastai/imagenet-fast/blob/master/imagenet_nv/distributed.py
import os, sys
import math
import time
import subprocess
import argparse
import torch
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.autograd import Variable
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from TTS.utils.generic_utils import load_config, create_experiment_folder


class DistributedSampler(Sampler):
    """
    Non shuffling Distributed Sampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None):
        super(DistributedSampler, self).__init__(dataset)
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


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
    dist.init_process_group(
        dist_backend,
        init_method=dist_url,
        world_size=num_gpus,
        rank=rank,
        group_name=group_name)


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
                for buf, synced in zip(
                        grads, _unflatten_dense_tensors(coalesced, grads)):
                    buf.copy_(synced)

    for param in list(module.parameters()):

        def allreduce_hook(*_):
            Variable._execution_engine.queue_callback(allreduce_params)

        if param.requires_grad:
            param.register_hook(allreduce_hook)

    def set_needs_reduction(self, *_):
        self.needs_reduction = True

    module.register_forward_hook(set_needs_reduction)
    return module


def main():
    """
    Call train.py as a new process and pass command arguments
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--continue_path',
        type=str,
        help='Training output folder to continue training. Use to continue a training. If it is used, "config_path" is ignored.',
        default='',
        required='--config_path' not in sys.argv)
    parser.add_argument(
        '--restore_path',
        type=str,
        help='Model file to be restored. Use to finetune a model.',
        default='')
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to config file for training.',
        required='--continue_path' not in sys.argv
    )
    args = parser.parse_args()

    # OUT_PATH = create_experiment_folder(CONFIG.output_path, CONFIG.run_name,
                                        # True)
    # stdout_path = os.path.join(OUT_PATH, "process_stdout/")

    num_gpus = torch.cuda.device_count()
    group_id = time.strftime("%Y_%m_%d-%H%M%S")

    # set arguments for train.py
    command = ['train.py']
    command.append('--continue_path={}'.format(args.continue_path))
    command.append('--restore_path={}'.format(args.restore_path))
    command.append('--config_path={}'.format(args.config_path))
    command.append('--group_id=group_{}'.format(group_id))
    command.append('')

    # run processes
    processes = []
    for i in range(num_gpus):
        my_env = os.environ.copy()
        my_env["PYTHON_EGG_CACHE"] = "/tmp/tmp{}".format(i)
        command[-1] = '--rank={}'.format(i)
        stdout = None if i == 0 else open(os.devnull, 'w')
        p = subprocess.Popen(['python3'] + command, stdout=stdout, env=my_env)
        processes.append(p)
        print(command)

    for p in processes:
        p.wait()


if __name__ == '__main__':
    main()
