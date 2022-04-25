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



if __name__ == "__main__":
    train_path = "/home/yjt/Documents/16833/sfmnet/runtime/train/2022_04_24_12_52_12.csv"
    content = pd.read_csv(train_path)
    loss = content["loss"].to_numpy()
    val_loss = content["val_loss"].to_numpy()
    epoch = content["epoch"].to_numpy()
    # print(epoch.to_numpy())
    plt.plot(epoch, loss)
    plt.plot(epoch, val_loss)
    plt.xlabel("Epoch")
    plt.ylabel("MSELoss")
    plt.title("Training Loss")
    plt.legend(("Loss of train set", "Loss of valid set"))
    plt.savefig("TrainingLoss.png")
