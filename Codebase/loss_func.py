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


class Sobel(nn.Module):
    def __init__(self):
        super().__init__()
        self.filter = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, stride=1, padding=0, bias=False)
        Gx = torch.tensor([[2.0, 0.0, -2.0], [4.0, 0.0, -4.0], [2.0, 0.0, -2.0]])
        Gy = torch.tensor([[2.0, 4.0, 2.0], [0.0, 0.0, 0.0], [-2.0, -4.0, -2.0]])
        G = torch.cat([Gx.unsqueeze(0), Gy.unsqueeze(0)], 0)
        G = G.unsqueeze(1)
        self.filter.weight = nn.Parameter(G, requires_grad=False)

    def forward(self, img):
        x = self.filter(img)
        # x = torch.mul(x, x)
        # x = torch.sum(x, dim=1, keepdim=True)
        # x = torch.sqrt(x)
        return x


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
    x1_t = sample(x1, warp, mode='bilinear', padding_mode='zeros')
    return mse_loss(x0, x1_t)


def spatial_smoothness_loss(x, order=1):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sobel_filter = Sobel().to(device)
    print(x.shape)
    b, c, h, w = x.shape
    x = x.reshape(b, 1, c*h, w)
    gradients = sobel_filter(x)
    print("shape of gradients:", gradients.shape)
    for i in range(order - 1):
        # gradients = torch.reshape(gradients, [b, h, w, -1])
        b, c, h, w = gradients.shape
        gradients = gradients.reshape(b, 1, c*h, w)
        gradients = sobel_filter(gradients)

    return torch.mean(torch.square(gradients))


if __name__ == "__main__":
    depth = torch.randn((3, 200, 200, 3))
    points = torch.randn((3, 200, 200, 3))
    pc_t = torch.randn((3, 200, 200, 3))
    result = forward_backward_consistency(depth, points, pc_t)

