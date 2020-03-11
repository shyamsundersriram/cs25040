import sys
import torch
import argparse
import numpy as np
from PIL import Image
import json
import random
from scipy.misc import toimage, imsave

import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.autograd import Variable
from torch.utils import data
import torchvision.transforms as transforms

from losses.loss import *
from nets.classifier import FCClassifier

from data.loader import PascalVOC
import torch.optim as optim
from utils import *
import pathlib

your_path = str(pathlib.Path(__file__).parent.absolute()) 


def train(dataset, model, optimizer, epoch):
    """
    TODO: Implement training for simple FC classifier.
        Input: Z-dimensional vector
        Output: label.
    """
    loss_function = nn.CrossEntropyLoss()
    batch_size = 1000
    print_every = 100 
    data_x, data_y = dataset
    _, counts_elements = np.unique(data_y, return_counts=True)
    totals = counts_elements / np.sum(counts_elements)
    weights = torch.Tensor(1 / totals) # weighing classes to eliminate bias
    tnsr_x = torch.from_numpy(data_x).type(torch.FloatTensor)
    tnsr_y = torch.from_numpy(data_y).type(torch.LongTensor)
    tnsr_data = data.TensorDataset(tnsr_x, tnsr_y)
    loader = data.DataLoader(tnsr_data, batch_size=batch_size, shuffle=True)
    model.train()

    for t, (x, y) in enumerate(loader):
        optimizer.zero_grad()
        pred = model(x)
        loss = cross_entropy1d(pred, y, weights)
        loss.backward()
        optimizer.step()

    torch.save(model, your_path + "/models/fc_cls.pkl")


def main():

    classifier = FCClassifier().float()

    optimizer = optim.Adam(classifier.parameters(), lr=1e-3) # pick an optimizer.

    dataset_x = np.load("./features/feats_x.npy")
    dataset_y = np.load("./features/feats_y.npy")

    num_epochs = 15

    for epoch in range(num_epochs):
        train([dataset_x, dataset_y], classifier, optimizer, epoch)


if __name__ == '__main__':
    main()
