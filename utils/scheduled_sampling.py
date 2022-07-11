import numpy as np
from numpy.random import random


class ScheduledSampler:
    def __init__(self, error_threshold=0.0, min_value=0.0, max_seq=-1):
        """
        Scheduled Sampling, samples wetter the predicted output should be used as input
        From "Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks"
        by Samy Bengio and Oriol Vinyals and Navdeep Jaitly and Noam Shazeer
        :param error_threshold: Error threshold for when a self.step() increases the iteration
        :param min_value: Minimum probability of using the output as input
        :param max_seq: Maximum steps of a sequence of using the output as input
        """
        super(ScheduledSampler, self).__init__()

        self.iteration = 0
        self.error_threshold = error_threshold
        self.min_value = min_value
        self.max_seq = max_seq if max_seq > 0 else np.inf

        self.seq_len = 0

    def __probability__(self, iteration: int = -1):
        """
        Probability function of using the output as input
        :param iteration: Number of the iteration, uses current iteration if iteration < 0
        :return: Probability 0..1
        """
        raise NotImplementedError('Implement this function')

    def probability(self, iteration: int = -1):
        """
        Post-processed probability of using the output as input
        :param iteration: Number of the iteration, uses current iteration if iteration < 0
        :return: Probability 0..1
        """
        iteration = iteration if iteration >= 0 else self.iteration
        return max(self.__probability__(iteration), self.min_value)

    def step(self, iteration=-1, error=0.0):
        """
        Increments the iteration if error < self.error_threshold
        :param iteration: Optionally sets the iteration explicitly
        :param error: Error, where the iteration is only increased when error < self.error_threshold
        """
        if error < self.error_threshold or self.error_threshold <= 0:
            self.iteration += 1

        self.iteration = iteration if iteration >= 0 else self.iteration

    def sample(self):
        """
        Samples wetter the output should be used as input
        :return: True if output should be used as input
        """
        sample = self.seq_len < self.max_seq and random() > self.probability()
        self.seq_len = self.seq_len + 1 if sample else 0
        return sample

    def sample_smooth(self):
        """
        Samples how much the output vs the input is used as new input
        :return: Ratio p=0..1 with input = p * output + (1 - p) * input
        """
        sample = self.probability() if self.seq_len < self.max_seq else 1.0
        self.seq_len = self.seq_len + 1 if sample < 0 else 0
        return sample


class LinearSampler(ScheduledSampler):
    def __init__(self, slope: float, error_threshold=0.0, min_value=0.0, max_seq=-1):
        """
        Samples with a linear slope wetter the predicted output should be used as input
        :param slope: Linear slope regarding the iterations
        :param error_threshold: Error threshold for increasing iteration with self.step()
        :param min_value: Minimum probability of using the output as input
        :param max_seq: Maximum steps of a sequence where the output is used as input
        """
        super(LinearSampler, self).__init__(error_threshold, min_value, max_seq)

        self.slope = slope

    def __probability__(self, iteration: int = -1):
        return 1 - iteration * self.slope


class ExponentialSampler(ScheduledSampler):
    def __init__(self, initial_probability: float, error_threshold=0.0, min_value=0.0, max_seq=-1):
        """
        Samples an exponential decrease, with probability = initial_probability ^ iteration
        :param initial_probability: Initial base, must be (0, 1]
        :param error_threshold: Error threshold for increasing iteration with self.step()
        :param min_value: Minimum probability of using the output as input
        :param max_seq: Maximum iterations in a closed loop sequence
        """
        super(ExponentialSampler, self).__init__(error_threshold, min_value, max_seq)

        assert(0 < initial_probability <= 1)
        self.initial_propability = initial_probability

    def __probability__(self, iteration: int = -1):
        return self.initial_propability**iteration
