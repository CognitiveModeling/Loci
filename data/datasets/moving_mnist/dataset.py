"""
This file contains the organization of the custom dataset such that it can be
read efficiently in combination with the DataLoader from PyTorch to prevent that
data reading and preparing becomes the bottleneck.

This script was inspired by
https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
"""

from torch.utils import data
import numpy as np
import h5py
import os
import mnist
import itertools
import math
from PIL import Image


__author__ = "Manuel Traub"

class MovingMNISTDataset(data.Dataset):
    def __init__(self,  numbers_per_image, mode, width, height, sequence_length):

        self.numbers_per_image = numbers_per_image
        self.sequence_length = sequence_length
        self.width = width
        self.height = height

        # Get the data and compute the possible combinations of numbers in one
        # sample
        self.mnist_dict, self.mnist_sizes = self.load_dataset(mode=mode)
        self.combinations = list(
            itertools.combinations(list(self.mnist_dict.keys()),
                                   numbers_per_image)
        )

    def __len__(self):
        return 100000000

    def __getitem__(self, index: int):
        """
        Single moving MNIST sample generation using the parameters of this
        simulator.
        :param combination_index: The index of the self.combination sample
        :return: The generated sample as numpy array(t, x, y)
        """

        # Determine the number-combination for the current sample
        combination = self.combinations[np.random.randint(len(self.combinations))]
        mnist_images = self.get_random_images(self.mnist_dict, self.mnist_sizes, combination)

        # randomly generate direc/speed/position, calculate velocity vector
        dx = [np.random.randint(1, 5) * np.random.choice([-1,1]) for i in range(len(mnist_images))]
        dy = [np.random.randint(1, 5) * np.random.choice([-1,1]) for i in range(len(mnist_images))]

        lx = [mnist_images[i].shape[0] for i in range(len(mnist_images))]
        ly = [mnist_images[i].shape[1] for i in range(len(mnist_images))]

        x = [np.random.randint(0, self.width - lx[i]) for i in range(len(mnist_images))]
        y = [np.random.randint(0, self.width - ly[i]) for i in range(len(mnist_images))]

        video = np.zeros((self.sequence_length, 3, self.width, self.height), dtype=np.float32)

        for t in range(self.sequence_length):

            for i in range(self.numbers_per_image):
                x[i] = x[i] + dx[i]
                y[i] = y[i] + dy[i]

                if x[i] < 0:
                    x[i] = -x[i]
                    dx[i] = -dx[i]

                if x[i] > self.width - lx[i] - 1:
                    x[i] = (self.width - lx[i] - 1) - (x[i] - (self.width - lx[i] - 1))
                    dx[i] = -dx[i]

                if y[i] < 0:
                    y[i] = -y[i]
                    dy[i] = -dy[i]

                if y[i] > self.height - ly[i] - 1:
                    y[i] = (self.height - ly[i] - 1) - (y[i] - (self.height - ly[i] - 1))
                    dy[i] = -dy[i]

                video[t,:,x[i]:x[i]+lx[i],y[i]:y[i]+ly[i]] += mnist_images[i]

        return (np.clip(video / 255.0, 0, 1).astype(np.float32), np.zeros((1, 3, self.width, self.height), dtype=np.float32))


    def load_dataset(self, mode="train"):
        """
        Loads the dataset using the python mnist package.
        :param mode: Any of "train" or "test"
        :return: Dictionary of MNIST numbers and a list of the pixel sizes
        """
        print("Loading Mnist {}".format(mode))
        if mode == "train" or mode == "val":
            mnist_images = mnist.train_images()
            mnist_labels = mnist.train_labels()
        elif mode == "test":
            mnist_images = mnist.test_images()
            mnist_labels = mnist.test_labels()

        n_labels = np.unique(mnist_labels)
        mnist_dict = {}
        mnist_sizes = []
        for i in n_labels:
            idxs = np.where(mnist_labels == i)
            mnist_dict[i] = mnist_images[idxs]
            mnist_sizes.append(mnist_dict[i].shape[0])

        return mnist_dict, mnist_sizes

    def get_random_images(self, dataset, size_list, id_list):
        """
        Returns a list of randomly chosen images from the given dataset.
        :param dataset: dictionary of images
        :param size_list: Corresponding sizes of the dataset images
        :param id_list: Numbers to put into the sample (e.g. [2, 7])
        :return: A list of randomly chosen images of the specified numbers
        """
        images = []
        for id in id_list:
            idx = np.random.randint(0, size_list[id])
            images.append(self.crop_image(dataset[id][idx]))

        return images

    def crop_image(self, img):
        sx = 0
        sy = 0
        ex = 28
        ey = 28

        while np.sum(img[sx,:]) == 0:
            sx += 1

        while np.sum(img[ex-1,:]) == 0:
            ex -= 1

        while np.sum(img[:,sy]) == 0:
            sy += 1

        while np.sum(img[:,ey-1]) == 0:
            ey -= 1

        return img[sx:ex,sy:ey]
