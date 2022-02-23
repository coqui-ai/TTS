import torch
from torch.utils.data.distributed import DistributedSampler

class DistributedSamplerWrapper(DistributedSampler):
    """ Wrapper over Sampler for distributed training. It allows you to use any sampler in distributed mode.
    It is especially useful in conjunction with torch.nn.parallel.DistributedDataParallel. In such a case, each
    process can pass a torch.utils.data.DistributedSampler instance as a torch.utils.data.DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note:
        Dataset is assumed to be of constant size.

    Args:
        sampler: Sampler used for subsampling.
        num_replicas (int, optional): Number of processes participating in distributed training. By default,
            world_size is retrieved from the current distributed group.
        rank (int, optional): Rank of the current process within num_replicas. By default, rank is retrieved
            from the current distributed group.
        shuffle (bool, optional): If True, sampler will shuffle the indices. Default: True.
        seed (int, optional): random seed used to shuffle the sampler if shuffle=True. This number should be
            identical across all processes in the distributed group. Default: 0.

    Reference: https://github.com/pytorch/pytorch/issues/23430

    """

    def __init__(
        self,
        sampler,
        num_replicas: int = None,
        rank: int = None,
        shuffle: bool = True,
        seed: int = 0
    ):
        super().__init__(
            sampler,
            num_replicas=num_replicas,
            rank=rank,
            shuffle=shuffle,
            seed=seed
        )

    def __iter__(self):
        indices = list(self.dataset)[:self.total_size]

        # Add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size , f"{len(indices)} != {self.total_size}"

        # Subsample
        offset = self.num_samples * self.rank
        indices = indices[offset : offset + self.num_samples]
        assert len(indices) == self.num_samples, f"{len(indices)} != {self.num_samples}"

        return iter(indices)

    def set_epoch(self, epoch):
        super().set_epoch(epoch)
        if hasattr(self.dataset, 'set_epoch'):
            self.dataset.set_epoch(epoch)
        elif hasattr(self.dataset, 'generator'):
            self.dataset.generator = torch.Generator().manual_seed(self.seed + epoch)

    def state_dict(self):
        return self.dataset.state_dict()

    def load_state_dict(self, state_dict):
        self.dataset.load_state_dict(state_dict)
