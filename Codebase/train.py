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
from MotionNet import MotionNet

import datetime
import pandas as pd

import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset,DataLoader,random_split 
from torch.nn.functional import mse_loss



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=8,
                                            help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=50,
                                            help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-5,
                                            help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                                            help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                                            help='seed for random initialisation')
    parser.add_argument('--load_mode', type=bool, default=False,
                                            help='load the existing model')
    args = parser.parse_args()
    train(args)


def apply(model, criterion, batch, targets, lengths):
    pred = model(torch.autograd.Variable(batch), lengths.cpu().numpy())
    loss = criterion(pred, torch.autograd.Variable(targets))
    return pred, loss



class Mask_L1Loss(nn.Module):
    def __init__(self):
        super(Mask_L1Loss, self).__init__()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        mask = target > 0
        # mask = mask.float()

        print(mask.requires_grad)
        plt.imshow(mask[0, 0, :].cpu().detach().numpy())
        plt.show()
        plt.imshow(input[0, 0, :].cpu().detach().numpy())
        plt.show()
        print(input[0, 0, 0, 0].cpu().detach().numpy())
        print(input.dtype)

        plt.imshow(target[0, 0, :].cpu().detach().numpy())
        plt.show()
        result = input*mask
        plt.imshow(result[0, 0, :].cpu().detach().numpy())
        plt.show()

        return mse_loss(input*mask, target, reduction='mean')


def train_model(model, optimizer, dl_train, dl_valid, batch_size, max_epochs, device):
    # criterion = nn.MSELoss(reduction='mean')
    criterion = Mask_L1Loss()
    metric_name = 'MSE'
    dfhistory = pd.DataFrame(columns = ["epoch","loss","val_loss"])
    for epoch in range(max_epochs):
        loss_sum = 0
        log_step_freq = 100
        step = 0
        for features,labels in dl_train:
            # labels = torch.zeros_like(features)
            step += 1
            optimizer.zero_grad()
            predictions = model(features, features, step)
            # print(predictions.dtype)
            # print(predictions)
            # print(labels)

            loss = 0
            # loss = criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
   
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.8f") %
                    (step, loss_sum/((step)*batch_size)))
            

                    
        # model.eval()
        val_loss_sum = 0.0
        val_step = 0
        for features,labels in dl_valid:
            val_step += 1
            with torch.no_grad():
                predictions = model(features)
                val_loss = criterion(predictions,labels)
 
            val_loss_sum += val_loss.item()
            
  
        info = (epoch, loss_sum/(batch_size*step), val_loss_sum/(batch_size*val_step))
        dfhistory.loc[epoch] = info
        
        print(("\nEPOCH = %d, loss = %.8f, val_loss = %.8f, ") 
            %info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
                
    print('Finished Training...')
    time_finished = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dfhistory.to_csv("/home/yjt/Documents/16833/sfmnet/runtime/train/"+time_finished+'.csv')
    torch.save(model.state_dict(), "/home/yjt/Documents/16833/sfmnet/runtime/model/"+time_finished+'.pkl')
    return model


def evaluate_test_set(model, dl_test):
    model.eval()
    criterion = nn.MSELoss()
    test_loss_sum = 0.0
    test_metric_sum = 0.0
    test_step = 1
    for test_step, (features,labels) in enumerate(dl_test, 1):
        with torch.no_grad():
            predictions = model(features)
            plt.imshow(predictions[0, 0,:].cpu().detach().numpy())
            plt.show()
            plt.imshow(labels[0, 0, :].cpu().detach().numpy())
            plt.show()
            test_loss = criterion(predictions,labels)
            test_metric = criterion(predictions,labels)

        test_loss_sum += test_loss.item()
        test_metric_sum += test_metric.item()
    print("test loss is:", test_loss_sum/test_step)


def train(args):
    random.seed(args.seed)
    datapath = '/mnt/back_data/Kitti/'
    model_path = '/home/yjt/Documents/16833/sfmnet/runtime/model/2022_04_22_20_45_30.pkl'

    KittiDataset = kitti_depth(datapath)
    n_train = int(len(KittiDataset)*0.95)
    n_valid = len(KittiDataset) - n_train
    ds_train,ds_valid = random_split(KittiDataset,[n_train,n_valid])
    dl_train,dl_valid = DataLoader(ds_train,batch_size = args.batch_size),DataLoader(ds_valid,batch_size = args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training using device: ", device)

    # model = StructureNet()
    model = MotionNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = model.to(device)

    if (args.load_mode == True):
        model.load_state_dict(torch.load(model_path))

    model = train_model(model, optimizer, dl_train, dl_valid, args.batch_size, args.num_epochs, device)

    evaluate_test_set(model, dl_train)


if __name__ == '__main__':
    main()