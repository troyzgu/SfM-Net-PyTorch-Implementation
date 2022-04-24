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


class StructureNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.cd_net = ConvDeconvNet()  # ConvDeconvNet should return 32 channels
        self.depth = nn.Conv2d(in_channels=64, 
                               out_channels=1, 
                               kernel_size=1)
        # self.grid = self.depth_to_point()

    
    def forward(self, x):
        x, _ = self.cd_net(x)
        depth = self.depth(x)
        depth = F.relu(depth)
        pc = depth_to_point(depth)
        return depth, pc
    
def depth_to_point(depth, camera_intrinsics=(0.5, 0.5, 1.0)):
    cx, cy, cf = camera_intrinsics
    b, c, h, w = depth.shape # what is the last dimensin c?

    x_l = torch.from_numpy(np.linspace(-cx, 1 - cx, h) / cf)
    y_l = torch.from_numpy(np.linspace(-cy, 1 - cy, w) / cf)

    x, y = torch.meshgrid(x_l, y_l)
    f = torch.ones_like(x)

    grid = torch.stack([x, y, f], 0).unsqueeze(0)
    grid = grid.repeat(b, 1, 1, 1).to("cuda")

    return depth * grid
    
