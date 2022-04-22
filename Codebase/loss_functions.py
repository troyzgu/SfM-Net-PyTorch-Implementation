import torch
from torch.nn import MSELoss as MSE
from torch.nn.functional import grid_sample as sample
import numpy as np



def warp_coordinates(points):
    b h w c = points.shape
    warp_x = points[:,:,:,0] * (w-1.0)
    warp_y = points[:,:,:,1] * (h-1.0)
    
    return torch.stack([warp_x, warp_y], dim = -1)

def forward_backward_consistency(d, points, pc_t):
    
    warp = warp_coordinates(points)
    d1_t = sample(d1, warp, , mode='bilinear', padding_mode='zeros')
    Z = pc_t[:, :, :, 2:3]
    
    return MSE(d1_t / 100, Z / 100)