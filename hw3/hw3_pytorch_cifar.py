import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import sampler

import torchvision.datasets as dset
import torchvision.transforms as T

import numpy as np
import os
import math

import matplotlib.pyplot as plt 

'''
In this section, you can experiment with a ConvNet architecture of your own
design.  Here, it is your job to experiment with architectures, hyperparameters,
loss functions, and optimizers to train a model.  Considering the training time,
especially on CPU, we only require you achieve at least 70% accuracy on the
CIFAR-10 validation set (loader_test in this hw3) within 10 epochs.
You must complete the test and train functions below.
You can use either nn.Module or nn.Sequential API during model design.
- Layers in torch.nn package: http://pytorch.org/docs/stable/nn.html
- Activations: http://pytorch.org/docs/stable/nn.html#non-linear-activations
- Loss functions: http://pytorch.org/docs/stable/nn.html#loss-functions
- Optimizers: http://pytorch.org/docs/stable/optim.html

To finish this section step by step, you need to:
(1) Prepare data by building dataset and dataloader. (alreadly provided below)
(2) Specify devices to train on (e.g. CPU or GPU). (alreadly provided below)
(3) Implement training code (6 points) & testing code (6 points) including
    saving and loading of models.
(4) Construct a model (12 points) and choose an optimizer (3 points).
(5) Describe what you did, any additional features that you implemented,
    and/or any graphs that you made in the process of training and
    evaluating your network.  Report final test accuracy @10 epochs in a
    writeup: hw3.pdf (3 points).
'''

'''
Data Preparation (NO need to modify):

(1) The torchvision.transforms package provides tools for preprocessing data
    and for performing data augmentation; here we set up a transform to
    preprocess the data by subtracting the mean RGB value and dividing by the
    standard deviation of each RGB value; we've hardcoded the mean and std.

(2) We set up a Dataset object for each split (train / val / test); Datasets
    load training examples one at a time, so we wrap each Dataset in a
    DataLoader which iterates through the Dataset and forms minibatches.  We
    divide the CIFAR-10 training set into train and val sets by passing a
    Sampler object to the DataLoader, telling how it should sample from the
    underlying Dataset.

(3) Note that, for the first time run, by seeting download as True, Pytorch
    will check the 'cifar_data' directory to decide if the CIFAR dataset needs
    to be downloaded.
'''
NUM_TRAIN = 49000
transform = T.Compose([
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
            ])
cifar10_train = dset.CIFAR10('./cifar_data', train=True, download=True,
                             transform=transform)
loader_train = DataLoader(cifar10_train, batch_size=64,
                          sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN)))
cifar10_val = dset.CIFAR10('./cifar_data', train=True, download=True,
                           transform=transform)
loader_val = DataLoader(cifar10_val, batch_size=64,
                        sampler=sampler.SubsetRandomSampler(range(NUM_TRAIN, 50000)))
cifar10_test = dset.CIFAR10('./cifar_data', train=False, download=True,
                            transform=transform)
loader_test = DataLoader(cifar10_test, batch_size=64)
PATH = './cifar_10trained.pth'
'''
Device specification (NO need to modify).
You have an option to use GPU by setting the flag to True below.

It is NOT necessary to use GPU for this assignment.  Note that if your computer does not have CUDA enabled,
torch.cuda.is_available() will return False and this notebook will fallback to
CPU mode.

The global variables dtype and device will control the data types throughout
this assignment.
'''
USE_GPU = False
dtype = torch.float32 # we will be using float throughout this tutorial
if USE_GPU and torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
# Constant to control how frequently we print train loss
print_every = 100
print('using device:', device)
best_acc = 0
'''
Training (6 points)
Train a model on CIFAR-10 using the PyTorch Module API.

Inputs:
- model: A PyTorch Module giving the model to train.
- optimizer: An Optimizer object we will use to train the model
- epochs: (Optional) A Python integer giving the number of epochs to train for

Returns: Nothing, but prints model accuracies during training.
'''
def train(model, optimizer, epochs=1):
    model = model.to(device=device)  # move the model parameters to CPU/GPU 
    #model.train()
    loss_fn = nn.CrossEntropyLoss() 
    losses = [] 
    accuracies = [] 
    for e in range(epochs):
        epoch_loss = []
        for t, (x, y) in enumerate(loader_train):
            # (1) put model to training mode
            # (2) move data to device, e.g. CPU or GPU
            # (3) forward and get loss
            # (4) Zero out all of the gradients for the variables which the optimizer
            # will update.
            # (5) the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            # (6)Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            ##########################################################################
            optimizer.zero_grad()
            output = model(x)
            loss = loss_fn(output, y)
            loss.backward() 
            optimizer.step() 
            epoch_loss.append(loss.item())
            ##########################################################################
            if t % print_every == 0:
                print('Epoch %d, Iteration %d, loss = %.4f' % (e, t, loss.item()))
                best_acc = test(loader_val, model)
                accuracies.append(best_acc)
            losses.append(np.mean(epoch_loss))
    torch.save(model.state_dict(), PATH)
    return losses, accuracies
'''
Testing (6 points)
Test a model on CIFAR-10 using the PyTorch Module API.

Inputs:
- loader:
- model: A PyTorch Module giving the model to test.

Returns: Nothing, but prints model accuracies during training.
'''
def test(loader, model):
    global best_acc
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    total = 0 
    correct = 0 
    accuracies = [] 
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for t, (x, y) in enumerate(loader):
            ##########################################################################
            output = model(x) 
            output, y = output.to(device), y.to(device)

            _, pred = torch.max(output.data, 1)
            total += y.size(0) 
            correct += pred.eq(y).sum().item()   
            # (1) move to device, e.g. CPU or GPU
            # (2) forward and calculate scores and predictions
            # (2) accumulate num_correct and num_samples
            ##########################################################################
        acc = float(correct / total)
        accuracies.append(acc)
        if loader.dataset.train and acc > best_acc:
            best_model = model.state_dict()
            ##########################################################################
            # (4)Save best model on validation set for final test
            ##########################################################################
            best_acc = acc
        print('Got %d / %d correct (%.2f)' % (correct, total, 100 * acc))
        return best_acc

##########################################################################
# TODO: YOUR CODE HERE
'''
Design/choose your own model structure (12 points) and optimizer (3 points).
Below are things you may want to try:

(1) Model hyperparams:
- Filter size, Number of filters and layers. You need to make careful choices to
tradeoff the computational efficiency and accuracy (especially in this assignment).
- Pooling vs Strided Convolution
- Batch normalization

(2) New Network architecture:
- You can borrow some cool ideas from novel convnets design, such as ResNet where
the input from the previous layer is added to the output
https://arxiv.org/abs/1512.03385
- Note: Don't directly use the existing network design.

(3) Different optimizers like SGD, Adam, Adagrad, RMSprop, etc.
**************************************************************************
# Basic Model and Optimizer.
Feel free to use more complicated ones to fit your model design.
class myNet(nn.Module):
    def __init__(self):
        super(myNet, self).__init__()
        # Set up your own convnets.

    def forward(self, x):
        # forward
        return out
model = myNet()
optimizer = optim.SGD(model.parameters(), lr, momentum, weight_decay)
# Describe your design details in the writeup hw3.pdf. (3 points)
**************************************************************************
Finish your model and optimizer below.
'''
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.conv1_bn = nn.BatchNorm2d(32)
        self.dropout1 = nn.Dropout2d(0.25)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 80, 5)
        self.conv2_bn = nn.BatchNorm2d(80)
        self.conv3 = nn.Conv2d(80, 64, 5)
        self.conv3_bn = nn.BatchNorm2d(64)
        self.dropout2 = nn.Dropout2d(0.25)
        self.fc1 = nn.Linear(24 * 24, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv1_bn(x)
        x = self.dropout1(x)
        x = self.pool(F.relu(x))
        x = self.conv2(x)
        x = self.conv2_bn(x)
        x = self.dropout2(x)
        x = self.conv3(x)
        x = self.conv3_bn(x)
        x = self.pool(F.relu(x))
        x = x.view(-1, 24 * 24) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x

'''
This is the writeup for the pytorch implementation. 

'''
##########################################################################

# You should get at least 70% accuracy
model = Net() 
optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.9)
losses, accuracies = train(model, optimizer, epochs=10)

##########################################################################

# load saved model to best_model for final testing
best_model = Net() 
best_model.load_state_dict(torch.load(PATH))
##########################################################################
test_accuracy = test(loader_test, best_model)
plt.plot(np.arange(len(accuracies)), np.array(accuracies)) 
plt.show() 
plt.plot(np.arange(len(losses)), np.array(losses))
plt.show() 
