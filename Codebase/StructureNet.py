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


class StructureNet(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.cd_net = ConvDeconvNet()  # ConvDeconvNet should return 32 channels
        self.depth = nn.Conv2D(in_channels=32, 
                               out_channels=1, 
                               kernel_size=1) 

    def forward(self, x):
        x, _ = self.cd_net(x)
        depth = self.depth(x)
        depth = F.relu(depth) * 99 + 1
        # pc = depth_to_point(depth)
        return depth
    
    def depth_to_point(depth, camera_intrinsics):
        pass
        return NotImplementedError
