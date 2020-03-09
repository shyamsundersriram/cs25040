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
    batch_size = 100 

    data_x, data_y = dataset

    model.train()

    for t, x in enumerate(data_x):
        y = data_y[t]
        if t % batch_size == 0: 
            optimizer.zero_grad()
            output = model(x)
            loss = cross_entropy2d(output, y)
            print(loss)
            #loss.backward() 
            optimizer.step() 
            #print('Epoch %d, Iteration %d, loss = %.4f' % (epoch, t, loss.item()))

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
