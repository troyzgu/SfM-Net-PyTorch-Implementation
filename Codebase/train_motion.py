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

warnings.filterwarnings("ignore")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16,
                                            help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=20,
                                            help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-6,
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



class TranslationLoss(nn.Module):
    def __init__(self):
        super(TranslationLoss, self).__init__()

    def forward(self, pred_R, pred_t, target_R, target_t) -> torch.Tensor:
        diff_t = (pred_t - target_t).unsqueeze(-1)
        loss_t = torch.norm(torch.bmm(torch.inverse(pred_R), diff_t))
        # loss_r = None
        error_r = torch.inverse(pred_R)@target_R
        mask = torch.zeros_like(error_r)
        mask[:, torch.arange(0,3), torch.arange(0,3) ] = 1.0
        error_r_trace = torch.sum(torch.bmm(mask, error_r), (1, 2))
        loss_r = torch.norm(torch.arccos(torch.min(torch.ones_like(error_r_trace), torch.max(-torch.ones_like(error_r_trace), (error_r_trace-1)/2))))
        # print(loss_t, loss_r)
        

        return loss_t + loss_r

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
        log_step_freq = 30
        step = 0
        for (frame0, frame1), (target_r, target_t) in dl_train:
            # labels = torch.zeros_like(features)
            step += 1
            optimizer.zero_grad()
            _, trans = model(frame0, frame1, step)
            cam_t, cam_p, cam_r = trans
            pred_r = r_mat(cam_r)
            pred_t = torch.reshape(cam_t, [frame0.shape[0], 3])
            loss = criterion(pred_r, pred_t, target_r, target_t)

            # loss = criterion(predictions,labels)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
   
            if step%log_step_freq == 0:
                r_mat(cam_r, True)
                r_print(pred_r, pred_t, target_r, target_t)  
                model(frame0, frame1, step, True)
                print(("[step = %d] loss: %.8f") %
                    (step, loss_sum/((step)*batch_size)))
                              
        # model.eval()
        val_loss_sum = 0.0
        val_step = 0
        for (frame0, frame1), (target_r, target_t) in dl_valid:
            val_step += 1
            with torch.no_grad():
                _, trans = model(frame0, frame1, step)
                cam_t, cam_p, cam_r = trans
                pred_r = r_mat(cam_r)
                pred_t = torch.reshape(cam_t, [frame0.shape[0], 3])
                val_loss = criterion(pred_r, pred_t, target_r, target_t)
            val_loss_sum += val_loss.item()
            
  
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


def evaluate_test_set(model, dl_test):
    model.eval()
    criterion = nn.MSELoss()
    test_loss_sum = 0.0
    test_metric_sum = 0.0
    test_step = 1
    for (frame0, frame1), (target_r, target_t) in dl_test:
        with torch.no_grad():
            mask = labels > 0
            predictions, _ = model(features, test_step)
            masked_pred = predictions * mask
            # file_name = "img_{}.png".format(test_step)
            # plt.imshow(masked_pred[0, 0,:].cpu().detach().numpy(), cmap='jet')
            # plt.axis('off')
            # plt.savefig("./img/{}".format(file_name), bbox_inches='tight')

            test_loss = criterion(predictions,labels)
            test_metric = criterion(predictions,labels)
            test_step += 1

        test_loss_sum += test_loss.item()
        test_metric_sum += test_metric.item()
    print("test loss is:", test_loss_sum/test_step)


def train(args):
    random.seed(args.seed)
    model_path = None

    KittiDataset = kitti_odom()
    n_train = int(len(KittiDataset)*0.95)
    n_valid = len(KittiDataset) - n_train
    ds_train,ds_valid = random_split(KittiDataset,[n_train,n_valid])
    dl_train,dl_valid = DataLoader(ds_train,batch_size = args.batch_size),DataLoader(ds_valid,batch_size = args.batch_size)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Training using device: ", device)

    model = MotionNet()
    # model = sfmnet()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    model = model.to(device)

    if (args.load_mode == True):
        model.load_state_dict(torch.load(model_path))

    model = train_model(model, optimizer, dl_train, dl_valid, args.batch_size, args.num_epochs, device)

    testdata = DataLoader(KittiDataset, batch_size = 1)
    evaluate_test_set(model, testdata)


if __name__ == '__main__':
    main()