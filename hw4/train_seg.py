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
from nets.zoomout import Zoomout
from nets.classifier import FCClassifier, DenseClassifier

from data.loader import *
import torch.optim as optim
from utils import *
import pathlib

your_path = str(pathlib.Path(__file__).parent.absolute()) 

def train(args, zoomout, model, train_loader, optimizer, epoch):
    count = 0

    for t, (x, y) in enumerate(train_loader):
        with torch.no_grad():
            zoom_feats = zoomout.forward(x.cpu().float())
        label = y 
        optimizer.zero_grad()
        predicts = model(zoom_feats)
        weights = torch.Tensor([ 2.4822, 41.2955, 55.9077, 34.6095, 46.5897, 41.7701, 46.5897, 28.3906,
            27.7405, 24.5541, 56.7812, 44.3171, 30.0331, 53.4412, 44.8642,  8.2217,
            44.3171, 57.6825, 39.0753, 43.7831, 43.7831]) # computed from train_cls weights. 
        loss = cross_entropy2d(predicts, label, weights)
        loss.backward(retain_graph=True)
        optimizer.step()
        print(loss.item())

        if t % 20 == 0:
            count = count + 1
            print("Epoch [%d/%d] Loss: %.4f" % (epoch+1, args.n_epoch, loss.item()))
    
            """
            Visualization of results.
            """
            pred = predicts[0,:,:,:]
            gt = y[0,:,:].data.numpy().squeeze()
            im = x[0,:,:,:].data.numpy().squeeze()
            im = np.swapaxes(im, 0, 2)
            im = np.swapaxes(im, 0, 1)
            _, pred_mx = torch.max(pred, 0)
            pred = pred_mx.data.numpy().squeeze()
            image = Image.fromarray(im.astype(np.uint8), mode='RGB')
            image.save("./imgs/im_" + str(count) + "_" + str(epoch) + "_.png")
            visualize("./lbls/pred_" + str(count) + "_" + str(epoch) + ".png", pred)
            visualize("./lbls/gt_" + str(count) + "_" + str(epoch) + ".png", gt)

    # Make sure to save your model periodically
    torch.save(model, your_path + "/models/full_model.pkl")

def val(args, zoomout, model, val_loader):
    # modified from https://github.com/wkentaro/pytorch-fcn/blob/master/examples/voc/evaluate.py
    model.eval()
    print("Validating...")
    label_trues, label_preds = [], []

    for batch_idx, (data, target) in enumerate(val_loader):

        data, target = data.float(), target.float()
        score = model(zoomout(data))

        _, pred = torch.max(score, 0)
        lbl_pred = pred.data.numpy().astype(np.int64)
        lbl_true = target.data.numpy().astype(np.int64)

        for _, lt, lp in zip(_, lbl_true, lbl_pred):
            label_trues.append(lt)
            label_preds.append(lp)

    n_class = 21
    metrics = label_accuracy_score(label_trues, label_preds, n_class=n_class)
    metrics = np.array(metrics)
    metrics *= 100
    print('''\
    Accuracy: {0}
    Accuracy Class: {1}
    Mean IU: {2}
    FWAV Accuracy: {3}'''.format(*metrics))


def main():
    # You can add any args you want here
    parser = argparse.ArgumentParser(description='Hyperparams')
    parser.add_argument('--model_path', nargs='?', type=str, default='zoomoutscratch_pascal_1_6.pkl', help='Path to the saved model')
    parser.add_argument('--n_epoch', nargs='?', type=int, default=10,    help='# of the epochs')
    parser.add_argument('--batch_size', nargs='?', type=int, default=2,  help='Batch Size')
    parser.add_argument('--l_rate', nargs='?', type=float, default=1e-4, help='Learning Rate')

    args = parser.parse_args()

    zoomout = Zoomout().float()

    # we will not train the feature extractor
    for param in zoomout.parameters():
        param.requires_grad = False


    fc_model_path = your_path + "/models/fc_cls.pkl"
    fc_classifier = torch.load(fc_model_path)
    classifier = DenseClassifier(fc_model=fc_classifier).float()

    """,
       TODO: Pick an optimizer.
       Reasonable optimizer: Adam with learning rate 1e-4.  Start in range [1e-3, 1e-4].
    """
    optimizer = optim.Adam(classifier.parameters(), lr=1e-3)

    dataset_train = PascalVOC(split = 'train')
    dataset_val = PascalVOC(split = 'val')

    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, num_workers=4)

    val_loader = torch.utils.data.DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4)

    for epoch in range(args.n_epoch):
        train(args, zoomout, classifier, train_loader, optimizer, epoch)
        val(args, zoomout, classifier, val_loader)

if __name__ == '__main__':
    main()
