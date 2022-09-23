"""
This file contains the organization of the custom dataset such that it can be
read efficiently in combination with the DataLoader from PyTorch to prevent that
data reading and preparing becomes the bottleneck.

This script was inspired by
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

from torch.utils import data
from typing import Tuple, Union, List
import numpy as np
import cv2
import os
import pickle

__author__ = "Manuel Traub"

class RamImage():
    def __init__(self, path):
        
        fd = open(path, 'rb')
        img_str = fd.read()
        fd.close()

        self.img_raw = np.frombuffer(img_str, np.uint8)

    def to_numpy(self):
        return cv2.imdecode(self.img_raw, cv2.IMREAD_COLOR) 


class VideoDataset(data.Dataset):
    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int], time: int, subset: bool = False):

        data_path = dataset_name if subset else f'data/data/video/{dataset_name}'
        data_path = os.path.join(root_path, data_path)
        self.file = os.path.join(data_path, f'dataset-{size[0]}x{size[1]}-{type}.pickle')

        if os.path.exists(self.file):
            self.load()
        else:
            data_path = os.path.join(data_path, "train")
            data_path = os.path.join(data_path, f'{size[0]}x{size[1]}')

            frames = []

            for file in os.listdir(data_path):
                if file.startswith("frame") and (file.endswith(".jpg") or file.endswith(".png")):
                    frames.append(os.path.join(data_path, file))

            frames.sort()

            if not subset:
                if type == "train":
                    frames = frames[:int(len(frames) * 0.9)]
                else:
                    frames = frames[int(len(frames) * 0.9):]

            num_samples = len(frames)

            self.imgs = []
            for i, path in enumerate(frames):
                self.imgs.append(RamImage(path))

                if not subset and i % 1000 == 0:
                    print(f"Loading Video {type} [{i * 100 / num_samples:.2f}]", flush=True)


            self.save()

        self.length = len(self.imgs) - time + 1
        self.time   = time
        self.size   = size

        if not subset:
            print(f'loaded {type} Video Dataset {dataset_name} [{self.length}]')

        if len(self) == 0:
            print(subset, self.length, len(self.frames), self.time)
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')

    def save(self):
        with open(self.file, "wb") as outfile:
    	    pickle.dump(self.imgs, outfile)

    def load(self):
        with open(self.file, "rb") as infile:
            self.imgs = pickle.load(infile)

    def __len__(self):
        return max(self.length, 0)

    def __getitem__(self, index: int):
        
        frames = []
        frames = np.zeros((self.time, 3, self.size[1], self.size[0]), dtype=np.float32)
        for i in range(self.time):
            img = self.imgs[index + i].to_numpy()
            frames[i] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return frames, np.zeros((1, 3, self.size[1], self.size[0]), dtype=np.float32)

class MultipleVideosDataset(data.Dataset):
    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int], time: int):

        data_path = f'data/data/video/{dataset_name}'
        data_path = os.path.join(root_path, data_path)
        self.train = (type == "train")

        self.datasets = []
        self.length = 0
        for dir in next(os.walk(data_path))[1]:
            if dir.startswith("00"): 
                self.datasets.append(VideoDataset(data_path, dir, type, size, time, True))
                self.length += self.datasets[-1].length
        
        self.background = None
        if "background.jpg" in os.listdir(data_path):
            self.background = cv2.imread(os.path.join(data_path, "background.jpg"))
            self.background = cv2.resize(self.background, dsize=size, interpolation=cv2.INTER_CUBIC)
            self.background = self.background.transpose(2, 0, 1).astype(float) / 255.0
            self.background = self.background.reshape(1, self.background.shape[0], self.background.shape[1], self.background.shape[2])

        print(f"MultipleVideosDataset: {self.length}")

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')

    def __len__(self):
        if self.train:
            return int(self.length * 0.9)

        return int(self.length * 0.1)

    def __getitem__(self, index: int):

        if not self.train:
            index += int((self.length / 300) * 0.9) * 300
        
        length = 0
        for dataset in self.datasets:
            length += len(dataset)

            if index < length:
                index = index - (length - len(dataset))
                if self.background is not None:
                    return dataset.__getitem__(int(index)), self.background

                return dataset.__getitem__(int(index))
