import os
import sys
import argparse
import time
import random
import pdb

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from DataKitti import kitti_depth
from StructureNet import StructureNet

import datetime
import pandas as pd

import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset,DataLoader,random_split 
from torch.nn.functional import mse_loss


def forward(input, target) :
        mask = target > 0
        print(mask.requires_grad)
        plt.imshow(mask[0, 0, :].cpu().detach().numpy())
        plt.show()
        plt.imshow(input[0, 0, :].cpu().detach().numpy())
        plt.show()

        plt.imshow(target[0, 0, :].cpu().detach().numpy())
        plt.show()
        result = input*mask
        plt.imshow(result[0, 0, :].cpu().detach().numpy())
        plt.show()
        return mse_loss(input*mask, target, reduction='mean')


if __name__ == "__main__":
    input = torch.zeros(16, 3, 120, 360, dtype = torch.float32)
    target = torch.rand(16, 3, 120, 360, dtype = torch.float32)*0.01
    loss = forward(input, target)
    print(loss)
