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

from DataKitti import kitti_odom
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
from transform import r_mat
import numpy as np

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                                            help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=100,
                                            help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-7,
                                            help='initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-9,
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

class TranslationLoss(nn.Module):
    def __init__(self):
        super(TranslationLoss, self).__init__()

    def forward(self, pred_R, pred_t, target_R, target_t) -> torch.Tensor:
        B, _, _ = pred_R.shape
        diff_t = (pred_t - target_t).unsqueeze(-1)
        # print(pred_t.shape, target_t.shape)
        loss_t = torch.norm(diff_t)
        # loss_t = torch.norm(torch.bmm(torch.inverse(pred_R), diff_t))
        # loss_r = torch.norm(pred_R-target_R)
        # print(pred_R)
        # print(target_R)

        error_r = torch.linalg.solve(pred_R, target_R) #torch.inverse(target_R)@pred_R
        error_r_trace = (torch.sum(torch.diagonal(error_r, dim1 = 1, dim2 = 2), -1) - 1)/2        
        loss_r = torch.norm(torch.arccos(torch.clamp(error_r_trace, min=-1.0, max=1.0)))
        # eye_mask = torch.ones(3, 3).float().to(device) - torch.eye(3).float().to(device)
        # mask = eye_mask.repeat(B, 1, 1)
        
        # diag_loss_r = torch.norm(torch.diagonal(target_R-pred_R, dim1 = 1, dim2 = 2))
        # other_loss_r = torch.norm(mask*(pred_R - target_R))
        # # print("diag loss and other entry loss:", diag_loss_r, other_loss_r)
        # loss_r = diag_loss_r + other_loss_r * 10

        return loss_t, loss_r*10

def r_print(pred_R, pred_t, target_R, target_t):
    print(pred_R[0, :])
    print(target_R[0, :])

def train_model(model, optimizer, dl_train, dl_valid, batch_size, max_epochs, device):
    # criterion = nn.MSELoss(reduction='mean')
    criterion = TranslationLoss()
    metric_name = 'MSE'
    dfhistory = pd.DataFrame(columns = ["epoch","loss","val_loss"])
    for epoch in range(max_epochs):
        loss_sum = 0
        log_step_freq = 50
        step = 0
        for (frame0, frame1), (target_r, target_t) in dl_train:
            # labels = torch.zeros_like(features)
            step += 1
            optimizer.zero_grad()
            _, trans = model(frame0, frame1, step)
            cam_t, cam_p, cam_r = trans
            pred_r = r_mat(cam_r)
            pred_t = torch.reshape(cam_t, [frame0.shape[0], 3])
            loss_t, loss_r = criterion(pred_r, pred_t, target_r, target_t)
            loss = loss_t + loss_r

            # loss = criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
   
            if step%log_step_freq == 0:
                print(("[step = %d] loss: %.8f") %
                    (step, loss_sum/((step)*batch_size)))
                print("loss_t is: {}, loss_r is: {}".format(loss_t, loss_r))
                              
        # model.eval()
        val_loss_sum = 0.0
        val_step = 1
        # for (frame0, frame1), (target_r, target_t) in dl_valid:
        #     with torch.no_grad():
        #         _, trans = model(frame0, frame1, step)
        #         cam_t, cam_p, cam_r = trans
        #         pred_r = r_mat(cam_r)
        #         pred_t = torch.reshape(cam_t, [frame0.shape[0], 3])
        #         val_loss = criterion(pred_r, pred_t, target_r, target_t)
        #     val_loss_sum += val_loss.item()
        #     val_step += 1
  
        info = (epoch, loss_sum/(batch_size*step), val_loss_sum/(batch_size*val_step))
        dfhistory.loc[epoch] = info
        print(("\nEPOCH = %d, loss = %.8f, val_loss = %.8f, ") % info)
        nowtime = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print("\n"+"=========="*8 + "%s"%nowtime)
                
    print('Finished Training...')
    time_finished = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    dfhistory.to_csv("/home/yjt/Documents/16833/sfmnet/runtime/train/"+time_finished+'.csv')
    torch.save(model.state_dict(), "/home/yjt/Documents/16833/sfmnet/runtime/model/"+time_finished+'.pkl')
    return model


def evaluate_test_set(model, dl_test, init_pose):
    model.eval()
    criterion = TranslationLoss()
    test_loss_sum = 0.0
    test_metric_sum = 0.0
    test_step = 1
    world_homo = init_pose
    all_data = init_pose[:3, :].reshape(-1)
    for (frame0, frame1), (target_r, target_t) in dl_test:
        with torch.no_grad():
            _, trans = model(frame0, frame1, test_step)
            cam_t, cam_p, cam_r = trans
            pred_r = r_mat(cam_r).reshape((3, 3))
            pred_t = torch.reshape(cam_t, (3, 1))
            pred_homo = torch.vstack((torch.hstack((pred_r.reshape((3,3)), pred_t.reshape((3, 1)))), torch.tensor([0, 0, 0, 1]).reshape(1, 4).to(device)))
            world_homo = pred_homo@world_homo
            if test_step % 50 == 0:
                print("*******prediction\n")
                print(pred_r)
                print(pred_t)
                print(target_r)
                print(target_t)
                print(world_homo)
            
            all_data = torch.vstack((all_data, world_homo[:3, :].reshape(-1)))
            test_step += 1

            # test_loss = criterion(pred_r, pred_t, target_r, target_t)
            # test_step += 1
    np.savetxt("/home/yjt/Documents/16833/kitti-odom-eval/output_00/00.txt", all_data.detach().cpu().numpy())
    #     test_loss_sum += test_loss.item()
    # print("test loss is:", test_loss_sum/test_step)


def train(args):
    random.seed(args.seed)
    # torch.random.seed(args.seed)
    model_path = "/home/yjt/Documents/16833/sfmnet/runtime/model/2022_05_03_20_46_52.pkl"

    KittiDataset = kitti_odom()
    init_pose = KittiDataset.get_initial_pose()
    n_train = int(len(KittiDataset))
    n_valid = len(KittiDataset) - n_train
    ds_train,ds_valid = random_split(KittiDataset,[n_train,n_valid])
    dl_train,dl_valid = DataLoader(ds_train,batch_size = args.batch_size),DataLoader(ds_valid,batch_size = args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training using device: ", device)

    model = MotionNet()
    # model = sfmnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9)
    model = model.to(device)

    if (args.load_mode == True):
        model.load_state_dict(torch.load(model_path))

    model = train_model(model, optimizer, dl_train, dl_valid, args.batch_size, args.num_epochs, device)

    testdata = DataLoader(KittiDataset, batch_size = 1)
    evaluate_test_set(model, testdata, init_pose)


if __name__ == '__main__':
    main()