import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import time
from datetime import datetime

import numpy as np
import os

from ConvDeconvSubnet import ConvDeconvNet


class MotionNet(nn.Module):
    def __init__(self, K=3):
        """
        param K: number of segmentation masks
        """
        super().__init__()
        self.num_masks = K

        self.cd_net = ConvDeconvNet()
        # We predict object masks from the image-sized feature map of the motion 
        # network using a 1 x 1 convolutional layer with sigmoid activations.
        self.obj_mask = nn.Conv2d(64, self.num_masks, 1) 

        self.d1 = nn.Linear(12 * 8 * 1024, 512) # input dimension is that of the flattened embedding layer
        self.d2 = nn.Linear(512, 512)

        self.cam_t = nn.Linear(512, 3) 
        self.cam_p = nn.Linear(512, 600) 
        self.cam_r = nn.Linear(512, 3) 

        self.obj_t = nn.Linear(512, 3 * self.num_masks)
        self.obj_p = nn.Linear(512, 600 * self.num_masks)
        self.obj_r = nn.Linear(512, 3 * self.num_masks)
        
    def forward(self, f0, f1, sharpness_multiplier):
        """
        param f0: frame 0
        param f1: frame 1
        param sharpness_multiplier: a parameter that is a function of 
            the number of step for which the network has been trained
        """
        x = torch.cat([f0, f1], -1) # depth-concatenate two frames 
        x, embedding = self.cd_net(x) # retrieve the embedding layer
        print(self.obj_mask(x).shape)
        # 1. object mask (predicted membership probability of each pixel to each rigid motion)
        obj_mask = F.sigmoid(self.obj_mask(x) * sharpness_multiplier)
        
        # Predict motion using the embedding layer
        # first implement two FC layers
        nbatch, *_ = embedding.shape  # nbatch x 12 x 4 x 1024


        embedding = torch.reshape(embedding, [nbatch, -1]) # flatten the layer except the batch
        embedding = self.d1(embedding)
        embedding = self.d2(embedding)
        
        # 2. object motion
        obj_t = self.obj_t(embedding)  # translation
        obj_p = self.obj_p(embedding)
        obj_p = torch.reshape(obj_p, [-1, self.num_masks, 600])
        obj_p = F.softmax(obj_p)  # pivot points
        obj_r = self.obj_r(embedding)  # angles of rotation
        
        # 3. camera pose
        cam_t = self.cam_t(embedding) # translation
        cam_p = self.cam_p(embedding) # pivot points
        cam_p = F.softmax(cam_p)
        cam_r = self.cam_r(embedding) # angles of rotation
        cam_r = F.tanh(cam_r)
        
        return (obj_mask, obj_t, obj_p, obj_r), (cam_t, cam_p, cam_r)
