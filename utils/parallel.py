import torch as th
from torch import distributed
import torch.multiprocessing as multiprocessing

from torch.utils import data
import os
import numpy as np


def run_parallel(fn, num_gpus=1, *args):
    """
    Initializes and starts the parallel training of a network
    :param fn: Function used for the training
    :param num_gpus: Number of GPUs
    :param args: Arguments for fn(rank, world_size, *args)
    """
    # Processes epochs parallel on GPUs
    multiprocessing.spawn(fn=_init_process, args=tuple([num_gpus, fn, *args]),
                          nprocs=num_gpus, join=True)


def _init_process(rank: int, world_size: int, fn, *args, backend='gloo'):
    """
    Initializes a process for one GPU
    :param rank: Number of the GPU (0..world_size-1)
    :param world_size: Number of GPUs available
    :param fn: Function used for the training
    :param args: Arguments for the function fn
    :param backend: Backend used by the framework
    """

    th.cuda.set_device(rank)

    def open_port(port=60000):
        os.environ['MASTER_ADDR'] = '127.0.0.1'
        master_port = port
        while True:
            try:
                os.environ['MASTER_PORT'] = f'{master_port}'
                distributed.init_process_group(backend=backend, rank=rank, world_size=world_size)
                return master_port
            except RuntimeError as e:
                if master_port >= port + 100:
                    raise e
                master_port = master_port + 1

    open_port()

    distributed.barrier()
    fn(rank, world_size, *args)


class DatasetPartition(data.Dataset):
    def __init__(self, parent: data.Dataset, rank: int, world_size: int):
        """
        Partition of Dataset for use in parallel training
        :param parent: Parent Dataset to be partitioned
        :param rank: Index [0..num-1] of the partition
        :param world_size: Number of all Dataset partitions
        """

        self.rank = rank
        self.world_size = world_size

        self.parent = parent
        self.indices = np.arange(len(parent))
        np.random.seed(123456)
        np.random.shuffle(self.indices)

        size = len(parent) // self.world_size
        if size == 0:
            raise ValueError(f'The datasets {parent} is empty')

        self.idx_start = self.rank * size
        self.idx_end = (self.rank + 1) * size

    def __len__(self):
        return self.idx_end - self.idx_start

    def __getitem__(self, idx: int):
        return self.parent[self.indices[self.idx_start + idx]]
