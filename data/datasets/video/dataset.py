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

__author__ = "Manuel Traub"

class VideoDataset(data.Dataset):
    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int], time: int, subset: bool = False):

        data_path = dataset_name if subset else f'data/data/video/{dataset_name}'
        data_path = os.path.join(root_path, data_path)
        self.data_path = data_path

        self.test_in_train = ("test" not in os.listdir(data_path))
        self.type = type

        if self.test_in_train or type == "train":
            data_path = os.path.join(data_path, "train")
        else:
            data_path = os.path.join(data_path, "test")

        data_path = os.path.join(data_path, f'{size[0]}x{size[1]}')

        self.frames = []
        self.backgrounds = []

        for file in os.listdir(data_path):
            if file.startswith("frame") and (file.endswith(".jpg") or file.endswith(".png")):
                self.frames.append(os.path.join(data_path, file))

            if file.startswith("background") and (file.endswith(".jpg") or file.endswith(".png")):
                self.backgrounds.append(os.path.join(data_path, file))

        self.frames.sort()
        self.backgrounds.sort()

        if not subset:
            if self.test_in_train and type == "train":
                self.frames = self.frames[:int(len(self.frames) * 0.9)]
                if len(self.backgrounds) > 1:
                    self.backgrounds = self.backgrounds[:int(len(self.backgrounds) * 0.9)]

            if self.test_in_train and type == "test":
                self.frames = self.frames[int(len(self.frames) * 0.9):]
                if len(self.backgrounds) > 1:
                    self.backgrounds = self.backgrounds[int(len(self.backgrounds) * 0.9):]

        self.length = len(self.frames) - time + 1
        self.time   = time
        self.size   = size

        if len(self) == 0:
            print(subset, self.length, len(self.frames), self.time)
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')

    def __len__(self):
        return max(self.length, 0)

    def __getitem__(self, index: int):
        
        """
        if self.backgrounds:
            frames = []
            backgrounds = []
            for i in range(self.time):
                img = cv2.imread(self.frames[index+i])
                bg  = None 
                if len(self.backgrounds) == 1:
                    bg = cv2.imread(self.backgrounds[0])
                else:
                    bg = cv2.imread(self.backgrounds[index+i])

                img = img.transpose(2, 0, 1).astype(float) / 255.0
                img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])

                bg = bg.transpose(2, 0, 1).astype(float) / 255.0
                bg = bg.reshape(1, bg.shape[0], bg.shape[1], bg.shape[2])

                frames.append(img)
                if len(self.backgrounds) > 1:
                    backgrounds.append(bg)

            if len(self.backgrounds) > 1:
                return np.concatenate(frames), np.concatenate(backgrounds)

            return np.concatenate(frames), bg
        """

        frames = []
        frames = np.zeros((self.time, 3, self.size[1], self.size[0]), dtype=np.float32)
        for i in range(self.time):
            img = cv2.imread(self.frames[index+i])
            img = img.transpose(2, 0, 1).astype(np.float32) / 255.0
            frames[i] = img

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

