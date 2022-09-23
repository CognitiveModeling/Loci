from torch.utils import data
from typing import Tuple, Union, List
import numpy as np
import json
import math
import cv2
import h5py
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

class ClevrerSample(data.Dataset):
    def __init__(self, root_path: str, data_path: str, size: Tuple[int, int]):

        data_path = os.path.join(root_path, data_path, "train", f'{size[0]}x{size[1]}')

        frames = []
        self.size = size

        for file in os.listdir(data_path):
            if file.startswith("frame") and (file.endswith(".jpg") or file.endswith(".png")):
                frames.append(os.path.join(data_path, file))

        frames.sort()
        self.imgs = []
        for path in frames:
            self.imgs.append(RamImage(path))

    def get_data(self):

        frames = np.zeros((128,3,self.size[1], self.size[0]),dtype=np.float32)
        for i in range(len(self.imgs)):
            img = self.imgs[i].to_numpy()
            frames[i] = img.transpose(2, 0, 1).astype(np.float32) / 255.0

        return frames


class ClevrerDataset(data.Dataset):

    def save(self):
        with open(self.file, "wb") as outfile:
    	    pickle.dump(self.samples, outfile)

    def load(self):
        with open(self.file, "rb") as infile:
            self.samples = pickle.load(infile)

    def __init__(self, root_path: str, dataset_name: str, type: str, size: Tuple[int, int]):

        data_path  = f'data/data/video/{dataset_name}'
        data_path  = os.path.join(root_path, data_path)
        self.file  = os.path.join(data_path, f'dataset-{size[0]}x{size[1]}.pickle')
        self.train = (type == "train")

        self.samples = []

        if os.path.exists(self.file):
            self.load()
        else:

            samples     = list(filter(lambda x: x.startswith("0"), next(os.walk(data_path))[1]))
            num_samples = len(samples)

            for i, dir in enumerate(samples):
                self.samples.append(ClevrerSample(data_path, dir, size))

                print(f"Loading CLEVRER [{i * 100 / num_samples:.2f}]", flush=True)

            self.save()
        
        self.length     = len(self.samples)
        self.background = None
        if "background.jpg" in os.listdir(data_path):
            self.background = cv2.imread(os.path.join(data_path, "background.jpg"))
            self.background = cv2.resize(self.background, dsize=size, interpolation=cv2.INTER_CUBIC)
            self.background = self.background.transpose(2, 0, 1).astype(np.float32) / 255.0
            self.background = self.background.reshape(1, self.background.shape[0], self.background.shape[1], self.background.shape[2])

        print(f"ClevrerDataset: {self.length}")

        if len(self) == 0:
            raise FileNotFoundError(f'Found no dataset at {self.data_path}')

    def __len__(self):
        if self.train:
            return int(self.length * 0.9)

        return int(self.length * 0.1)

    def __getitem__(self, index: int):

        if not self.train:
            index += int(self.length * 0.9)
        
        return (
            self.samples[index].get_data(),
            self.background,
        )
