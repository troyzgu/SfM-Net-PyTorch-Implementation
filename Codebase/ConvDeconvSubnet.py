import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2D(*args, **kwargs)
        self.batch_norm = nn.BatchNormalization2D(*args[-1])

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return F.relu(x)

class Deconv2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = nn.Conv2DTranspose(*args, **kwargs)
        self.batch_norm = nn.BatchNormalization2D(*args[-1])

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        return F.relu(x)

class ConvDeconvNet(nn.Module):
    def __init__(self):
        super().__init__()
        # convolution
        kwargs = {'kernel_size':3, 'padding':'same'}
        self.input_channels = 3 # RGB(3) or grayscale(1)
        
        self.c11 = Conv2D(self.input_channels, 32, ** kwargs)

        self.c21 = Conv2D(32, 64, strides=2, **kwargs)
        self.c22 = Conv2D(64, 64, **kwargs)

        self.c31 = Conv2D(64, 128, strides=2,  **kwargs)
        self.c32 = Conv2D(128, 128, **kwargs)

        self.c41 = Conv2D(128, 256, strides=2, **kwargs)
        self.c42 = Conv2D(256, 256, **kwargs)

        self.c51 = Conv2D(256, 512, strides=2, **kwargs)
        self.c52 = Conv2D(512, 512, **kwargs)

        self.c61 = Conv2D(512, 1024, strides=2, **kwargs)
        self.c62 = Conv2D(1024, 1024, **kwargs)

        # deconvlution
        kwargs = {'kernel_size': 3, 'strides': 2, 'padding': 'same'}

        self.u5 = Deconv2D(1024, 512, **kwargs)
        self.u4 = Deconv2D(512, 256, **kwargs)
        self.u3 = Deconv2D(256, 128, **kwargs)
        self.u2 = Deconv2D(128, 64, **kwargs)
        self.u1 = Deconv2D(64, 32, **kwargs)
        
    def forward(self, x):
        # convolution
        x1 = self.c11(x)

        x2 = self.c21(x1)
        x2 = self.c22(x2)

        x3 = self.c31(x2)
        x3 = self.c32(x3)

        x4 = self.c41(x3)
        x4 = self.c42(x4)

        x5 = self.c51(x4)
        x5 = self.c52(x5)

        x6 = self.c61(x5)
        embedding = self.c62(x6) # for motion network
        
        # deconvlution
        u5 = self.u5(embedding)
        u5 = torch.cat([x5, u5], -1)

        u4 = self.u4(u5)
        u4 = torch.cat([x4, u4], -1)

        u3 = self.u3(u4)
        u3 = torch.cat([x3, u3], -1)

        u2 = self.u2(u3)
        u2 = torch.cat([x2, u2], -1)

        u1 = self.u1(u2)
        u1 = torch.cat([x1, u1], -1)
        
        return u1, embedding
