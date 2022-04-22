import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.utils.data import Dataset, DataLoader,random_split
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score



import torch
from torch.nn import MSELoss as MSE
from torch.nn.functional import grid_sample as sample
from torch.nn.functional import mse_loss


def warp_coordinates(points):
    '''
    input:
        points Points from point cloud with (batch_size, height, width, channel)
    
    return:
        Points after warp
    '''
    b, h, w, c = points.shape
    warp_x = points[:,:,:,0] * (w-1.0)
    warp_y = points[:,:,:,1] * (h-1.0)
    
    return torch.stack([warp_x, warp_y], dim = -1)

def forward_backward_consistency(d, points, pc_t):
    '''
    input:
        d Depth of the frame, size is: (batch_size, height, width, channel)
        points Points from point cloud with (batch_size, height, width, channel)
        pc_t 
    '''
    
    warp = warp_coordinates(points)
    d1_t = sample(d, warp, mode='bilinear', padding_mode='zeros')
    Z = pc_t[:, :, :, 2:3]
    
    return mse_loss(d1_t / 100, Z / 100)


def frame_loss(x0, x1, points):
    warp = warp_coordinates(points)
    # warp = warp_image(x1, params)
    x1_t = sample(x1, warp, mode='bilinear', padding_mode='zeros')
    return nn.mse(x0, x1_t)


def spatial_smoothness_loss(x, order=1):
    b, h, w, c = x.shape
    gradients = torch.image.sobel_edges(x)
    for i in range(order - 1):
        gradients = torch.reshape(gradients, [b, h, w, -1])
        gradients = torch.image.sobel_edges(gradients)
    return torch.reduce_mean(torch.square(gradients))


if __name__ == "__main__":
    depth = torch.randn((3, 200, 200, 3))
    points = torch.randn((3, 200, 200, 3))
    pc_t = torch.randn((3, 200, 200, 3))
    result = forward_backward_consistency(depth, points, pc_t)

