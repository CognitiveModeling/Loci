import torch.nn as nn
import torch as th
import numpy as np
from einops import rearrange, repeat, reduce

from typing import Tuple, Union, List

__author__ = "Manuel Traub"

class DeviceSideDataset:

    def __init__(self, dataset, device, batch_size):

        self.batch_size = batch_size
        
        self.data = [[]]
        if isinstance(dataset[0], tuple) or isinstance(dataset[0], list):
            for i in range(1, len(dataset[0])):
                self.data.append([])

            for i in range(len(dataset)):
                item = dataset[i]
                for n in range(len(item)):
                    self.data[n].append(item[n])

                if i % 100 == 0:
                    print(f"loading data {i * 100 / len(dataset):.2f}%")

        else:
            for i in range(len(dataset)):
                self.data[0].append(dataset[i])
                if i % 100 == 0:
                    print(f"loading data {i * 100 / len(dataset):.2f}%")

        print(f"loading data {100:.2f}%")
        for i in range(len(self.data)):
            self.data[i] = th.tensor(np.stack(self.data[i])).float().to(device)
            print(f"pushed[{i}]: {self.data[i].element_size()} * {self.data[i].nelement()} = {self.data[i].element_size() * self.data[i].nelement()}")

        self.shuffle()

    def __len__(self):
        return self.data[0].shape[0] // self.batch_size

    def __iter__(self):
        self.shuffle()
        self.batch_counter = 0
        return self

    def next(self):
        return self.__next__()

    def __next__(self):
        batch_start = self.batch_counter * self.batch_size
        batch_end   = (self.batch_counter + 1) * self.batch_size
        self.batch_counter += 1

        if batch_end >= self.data[0].shape[0]:
            raise StopIteration

        batch = []
        for i in range(len(self.data)):
            batch.append(self.data[i][batch_start:batch_end])
        
        return tuple(batch)

    def shuffle(self):
        with th.no_grad():
            indices = th.randperm(self.data[0].shape[0])

            for i in range(len(self.data)):
                self.data[i] = self.data[i][indices]

