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
from sfmnet import sfmnet

import datetime
import pandas as pd

import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
from torch.utils.data import Dataset,DataLoader,random_split 
from torch.nn.functional import mse_loss
from loss_func import frame_loss, spatial_smoothness_loss, forward_backward_consistency
import warnings
import cv2
import numpy as np


warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                                            help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=600,
                                            help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-2,
                                            help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-10,
                                            help='weight_decay rate')
    parser.add_argument('--seed', type=int, default=123,
                                            help='seed for random initialisation')
    parser.add_argument('--load_mode', type=bool, default=False,
                                            help='load the existing model')
    parser.add_argument("--root_path", type = str, default = '/mnt/back_data/Kitti/')
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
        return mse_loss(input*mask, target, reduction='mean')

def total_loss(frame0, frame1, model, step):
    depth, points, objs, cams, motion_map, points_2d, flow = model(frame0, frame1, step)
    depth_inv, points_inv, objs_inv, cams_inv, motion_map_inv, points_2d_inv, flow_inv = model(frame1, frame0, step)

    frameloss = frame_loss(frame0, frame1, points_2d)
    frameloss_inv = frame_loss(frame1, frame0, points_2d_inv)

    smoothloss_depth = spatial_smoothness_loss(depth/100, 2)
    smoothloss_depth_inv = spatial_smoothness_loss(depth_inv/100, 2)

    smoothloss_flow = spatial_smoothness_loss(flow)
    smoothloss_flow_inv = spatial_smoothness_loss(flow_inv)

    b, h, w, k, c = motion_map.shape
    _motion_map = motion_map.reshape(b, h, w, k*c)
    _motion_map_inv = motion_map_inv.reshape(b, h, w, k*c)
    _motion_map = torch.movedim(_motion_map, -1, 1)
    _motion_map_inv = torch.movedim(_motion_map_inv, -1, 1)

    smoothloss_motion = spatial_smoothness_loss(_motion_map)
    smoothloss_motion_inv = spatial_smoothness_loss(_motion_map_inv)

    forward_backward_loss = forward_backward_consistency(depth_inv, points_2d, points)
    forward_backward_loss_inv = forward_backward_consistency(depth, points_2d_inv, points_inv)

    # loss = frameloss + frameloss_inv + smoothloss_depth + smoothloss_depth_inv + smoothloss_flow + smoothloss_flow_inv + \
    #     smoothloss_motion + smoothloss_motion_inv + forward_backward_loss + forward_backward_loss_inv

    loss = forward_backward_loss + forward_backward_loss_inv

    return loss


def train_model(model, optimizer, dl_train, dl_valid, batch_size, max_epochs, device):
    criterion = Mask_L1Loss()
    metric_name = 'MSE'
    dfhistory = pd.DataFrame(columns = ["epoch","loss","val_loss"])
    for epoch in range(max_epochs):
        loss_sum = 0
        log_step_freq = 1
        step = 0
        for features, labels in dl_train:
            step += 1
            optimizer.zero_grad()
            predictions,_ = model(features)
            loss = criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
   
            if step%log_step_freq == 0:   
                print(("[step = %d] loss: %.8f") %
                    (step, loss_sum/((step)*batch_size)))
            

                    
        model.eval()
        val_loss_sum = 0.0
        val_step = 0
        for features,labels in dl_valid:
            val_step += 1
            with torch.no_grad():
                predictions,_ = model(features)
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
            mask = labels > 0
            predictions, _ = model(features)
            masked_pred = predictions * mask
            file_name = "img_{}.png".format(test_step-1)
            tmp_im = (labels[0, 0,:].cpu().detach().numpy()*256).astype(np.uint8)
            cv2.imwrite("./img/{}".format(file_name), tmp_im)
            test_loss = criterion(predictions,labels)
            test_metric = criterion(predictions,labels)

        test_loss_sum += test_loss.item()
        test_metric_sum += test_metric.item()
    print("test loss is:", test_loss_sum/test_step)


def train(args):
    random.seed(args.seed)
    datapath = args.root_path
    model_path = '/home/yjt/Documents/16833/sfmnet/runtime/model/2022_04_24_12_52_12.pkl'

    KittiDataset = kitti_depth(datapath)
    n_train = int(len(KittiDataset)*0.95)
    n_valid = len(KittiDataset) - n_train
    ds_train,ds_valid = random_split(KittiDataset,[n_train,n_valid])
    dl_train,dl_valid = DataLoader(ds_train,batch_size = args.batch_size),DataLoader(ds_valid,batch_size = args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training using device: ", device)
    model = StructureNet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = model.to(device)

    if (args.load_mode == True):
        model.load_state_dict(torch.load(model_path))

    model = train_model(model, optimizer, dl_train, dl_valid, args.batch_size, args.num_epochs, device)

    testdata = DataLoader(KittiDataset, batch_size = 1)
    evaluate_test_set(model, testdata)


if __name__ == '__main__':
    main()