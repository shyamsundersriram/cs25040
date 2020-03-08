"""
TODO: Implement zoomout feature extractor.
"""

from data.loader import PascalVOC
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.models as models

class Zoomout(nn.Module):
    def __init__(self):
        super(Zoomout, self).__init__()

        # load the pre-trained ImageNet CNN and list out the layers
        self.vgg = models.vgg11(pretrained=True)
        self.feature_list = list(self.vgg.features.children())
        self.feature0 = self.vgg.features[:2]
        self.feature1 = self.vgg.features[:5]
        self.feature2 = self.vgg.features[:10]
        self.feature3 = self.vgg.features[:15]
        self.feature4 = self.vgg.features[:20]
        """
        TODO:  load the correct layers to extract zoomout features.
        """
    def forward(self, x):

        """
        TODO: load the correct layers to extract zoomout features.
        Hint: use F.upsample_bilinear and then torch.cat.
        """
        _, _, self.H, self.W = x.shape
        activation0 = F.interpolate(self.feature0(x), (self.H, self.W))
        activation1 = F.interpolate(self.feature1(x), (self.H, self.W))
        activation2 = F.interpolate(self.feature2(x), (self.H, self.W))
        activation3 = F.interpolate(self.feature3(x), (self.H, self.W))
        activation4 = F.interpolate(self.feature4(x), (self.H, self.W))
        self.activations = [activation0, activation1, activation2, activation3, activation4]
        hypermat = torch.cat(self.activations, dim=1)
        return hypermat 

def test_zoomout(): 
    dataset_train = PascalVOC(split ='train')
    image = dataset_train[1][0]
    print(Zoomout().forward(image).shape)


        
