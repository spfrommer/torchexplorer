from torchvision import transforms, datasets
from typing import *
import torch
import os
from torch.utils.data import Dataset

data_shape = {
    'mnist': (1, 28, 28),
    'cifar10': (3, 32, 32),
}
class_n = {
    'mnist': 10,
    'cifar10': 10,
}


def get_normalize_layer(dataset: str) -> torch.nn.Module:
    """Return the dataset's normalization layer"""
    if dataset == 'mnist':
        return NormalizeLayer(_MNIST_MEAN, _MNIST_STDDEV)
    elif dataset == 'cifar10':
        return NormalizeLayer(_CIFAR10_MEAN, _CIFAR10_STDDEV)


_MNIST_MEAN = [0.1307]
_MNIST_STDDEV = [0.3081]

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STDDEV = [0.2023, 0.1994, 0.2010]


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.

      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means: List[float], sds: List[float]):
        """
        :param means: the channel means
        :param sds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.means = torch.tensor(means).cuda()
        self.sds = torch.tensor(sds).cuda()

    def forward(self, input: torch.tensor):
        (batch_size, num_channels, height, width) = input.shape
        means = self.means.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        sds = self.sds.repeat((batch_size, height, width, 1)).permute(0, 3, 1, 2)
        return (input - means) / sds