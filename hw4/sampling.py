import sys
import torch
import numpy as np

from torch.utils import data

from nets.zoomout import Zoomout
from data.loader import PascalVOC
from utils import *
import gc
import random

def extract_samples(zoomout, dataset):
    """
    TODO: Follow the directions in the README
    to extract a dataset of 1x1xZ features along with their labels.
    Predict from zoomout using:
         with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
    """
    for image_idx in range(len(dataset)):
        images, labels = dataset[image_idx]
        max_label = torch.max(labels) #optimization of code 
        unique_label = torch.unique(labels)
        hw_samples = {} 
        for label in unique_label:
            indices = torch.where(labels == label)
            size = list(indices[0].size())
            if size[0] >= 3:
                hw_samples[label] = [] 
                random_samples = random.sample(range(size[0]), 3) 
                for randint in random_samples: 
                    hw_samples[label].append((indices[0][randint], indices[1][randint])) 
                print(hw_samples)
        with torch.no_grad():
            zoom_feats = zoomout(images.cpu().float().unsqueeze(0))
        hypercols = []     
        for new_label, final_indices in enumerate(hw_samples): 
            for (h, w) in final_indices: 
                hypercol = zoom_feats[h, w, :]
                hypercols.append(hypercol)
        return hypercols

















    return features, features_labels

def main():
    zoomout = Zoomout().cpu().float()
    for param in zoomout.parameters():
        param.requires_grad = False

    dataset_train = PascalVOC(split = 'train')

    features, labels = extract_samples(zoomout, dataset_train)

    np.save("./features/feats_x.npy", features)
    np.save("./features/feats_y.npy", labels)


if __name__ == '__main__':
    main()
