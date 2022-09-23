import os
from utils.configuration import Configuration
import time
import torch as th
import numpy as np
from einops import rearrange, repeat, reduce
import cv2

class Timer:
    
    def __init__(self):
        self.last   = time.time()
        self.passed = 0
        self.sum    = 0

    def __str__(self):
        self.passed = self.passed * 0.99 + time.time() - self.last
        self.sum    = self.sum * 0.99 + 1
        passed      = self.passed / self.sum
        self.last = time.time()

        if passed > 1:
            return f"{passed:.2f}s/it"

        return f"{1.0/passed:.2f}it/s"

class UEMA:
    
    def __init__(self, memory = 100):
        self.value  = 0
        self.sum    = 1e-30
        self.decay  = np.exp(-1 / memory)

    def update(self, value):
        self.value = self.value * self.decay + value
        self.sum   = self.sum   * self.decay + 1

    def __float__(self):
        return self.value / self.sum

class BinaryStatistics:
    
    def __init__(self):
        self.true_positive  = 0
        self.true_negative  = 0
        self.false_positive = 0
        self.false_negative = 0

    def update(self, outputs, labels):
        outputs = th.round(outputs)
        self.true_positive  += th.sum((outputs == labels).float() * (labels == th.ones_like(labels)).float()).item()
        self.true_negative  += th.sum((outputs == labels).float() * (labels == th.zeros_like(labels)).float()).item()
        self.false_positive += th.sum((outputs != labels).float() * (labels == th.zeros_like(labels)).float()).item()
        self.false_negative += th.sum((outputs != labels).float() * (labels == th.ones_like(labels)).float()).item()

    def accuracy(self):
        return 100 * (self.true_positive + self.true_negative) / (self.true_positive + self.true_negative + self.false_positive + self.false_negative + 1e-30)

    def sensitivity(self):
        return 100 * self.true_positive / (self.true_positive + self.false_negative + 1e-30)

    def specificity(self):
        return 100 * self.true_negative / (self.true_negative + self.false_positive + 1e-30)


def model_path(cfg: Configuration, overwrite=False, move_old=True):
    """
    Makes the model path, option to not overwrite
    :param cfg: Configuration file with the model path
    :param overwrite: Overwrites the files in the directory, else makes a new directory
    :param move_old: Moves old folder with the same name to an old folder, if not overwrite
    :return: Model path
    """
    _path = os.path.join('out')
    path = os.path.join(_path, cfg.model_path)

    if not os.path.exists(_path):
        os.makedirs(_path)

    if not overwrite:
        if move_old:
            # Moves existing directory to an old folder
            if os.path.exists(path):
                old_path = os.path.join(_path, f'{cfg.model_path}_old')
                if not os.path.exists(old_path):
                    os.makedirs(old_path)
                _old_path = os.path.join(old_path, cfg.model_path)
                i = 0
                while os.path.exists(_old_path):
                    i = i + 1
                    _old_path = os.path.join(old_path, f'{cfg.model_path}_{i}')
                os.renames(path, _old_path)
        else:
            # Increases number after directory name for each new path
            i = 0
            while os.path.exists(path):
                i = i + 1
                path = os.path.join(_path, f'{cfg.model_path}_{i}')

    return path
