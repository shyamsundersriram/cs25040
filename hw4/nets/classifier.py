"""
   Here you will implement a relatively shallow neural net classifier on top of the hypercolumn (zoomout) features.
   You can look at a sample MNIST classifier here: https://github.com/pytorch/examples/blob/master/mnist/main.py
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models
from nets.zoomout import *
import numpy as np
from torchvision import transforms

class FCClassifier(nn.Module):
    """
        Fully connected classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, n_classes=21):
        super(FCClassifier, self).__init__()
        """
        TODO: Implement a fully connected classifier.
        """
        self.fc1 = nn.Linear(1472, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 21)
        # You will need to compute these and store as *.npy files
        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))

    def forward(self, x):
        # normalization
        x = torch.Tensor(x)
        x = (x - self.mean)/ (self.std + 1e-3)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x 


class DenseClassifier(nn.Module):
    """
        Convolutional classifier on top of zoomout features.
        Input: extracted zoomout features.
        Output: H x W x 21 softmax probabilities.
    """
    def __init__(self, fc_model, n_classes=21):
        super(DenseClassifier, self).__init__()
        """
        TODO: Convert a fully connected classifier to 1x1 convolutional.
        """
        self.conv1 = nn.Conv2d(3, 32, 1)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        self.conv2 = nn.Conv2d(32, 80, 1)
        self.conv2_bn = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 64, 1)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(1472, 120)
        self.fc2 = nn.Linear(120, 64)
        self.fc3 = nn.Linear(64, 21)

        self.mean = torch.Tensor(np.load("./features/mean.npy"))
        self.std = torch.Tensor(np.load("./features/std.npy"))

        # You'll need to add these trailing dimensions so that it broadcasts correctly.
        self.mean = torch.Tensor(np.expand_dims(np.expand_dims(mean, -1), -1))
        self.std = torch.Tensor(np.expand_dims(np.expand_dims(std, -1), -1))

    def forward(self, x):
        """
        Make sure to upsample back to 224x224 --take a look at F.upsample_bilinear
        """

        # normalization
        x = (x - self.mean)/ (self.std + 1e-3)
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.dropout1(x)
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = x.view(-1, 1472) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x
