import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv = nn.Conv2d(*args, **kwargs)
        self.batch_norm = nn.BatchNorm2d(args[-1])

    def forward(self, x):
        x = self.conv(x)
        x = self.batch_norm(x)
        return F.relu(x)

class Deconv2D(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.conv_transpose = nn.ConvTranspose2d(*args, **kwargs)
        self.batch_norm = nn.BatchNorm2d(args[-1])

    def forward(self, x):
        x = self.conv_transpose(x)
        x = self.batch_norm(x)
        return F.relu(x)

class ConvDeconvNet(nn.Module):
    def __init__(self, input_channels = 3 ):
        super().__init__()
        # convolution
        kwargs = {'kernel_size':3, 'padding_mode':'zeros'}
        self.input_channels = input_channels # RGB(3) or grayscale(1)
        
        self.c11 = Conv2D(self.input_channels, 32, padding = 'same', **kwargs)

        self.c21 = Conv2D(32, 64, stride=2, padding = 1, **kwargs)
        self.c22 = Conv2D(64, 64, padding = 'same', **kwargs)

        self.c31 = Conv2D(64, 128, stride=2,  padding = 1, **kwargs)
        self.c32 = Conv2D(128, 128, padding = 'same', **kwargs)

        self.c41 = Conv2D(128, 256, stride=2, padding = 1, **kwargs)
        self.c42 = Conv2D(256, 256, padding = 'same', **kwargs)

        self.c51 = Conv2D(256, 512, stride=2, padding = 1, **kwargs)
        self.c52 = Conv2D(512, 512, padding = 'same', **kwargs)

        self.c61 = Conv2D(512, 1024, stride=2, padding = 1, **kwargs)
        self.c62 = Conv2D(1024, 1024, padding = 'same', **kwargs)

        # deconvlution
        kwargs = {'padding':1, 'output_padding': 1, 'kernel_size': 3, 'stride': 2}

        self.u5 = Deconv2D(1024, 512, **kwargs)
        self.u4 = Deconv2D(1024, 256, **kwargs)
        # self.upcov4 = 
        self.u3 = Deconv2D(512, 128, **kwargs)
        self.u2 = Deconv2D(256, 64, **kwargs)
        self.u1 = Deconv2D(128, 32, **kwargs)
        
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
        u5 = torch.cat([x5, u5], 1)

        u4 = self.u4(u5)
        u4 = torch.cat([x4, u4], 1)

        u3 = self.u3(u4)
        u3 = torch.cat([x3, u3], 1)

        u2 = self.u2(u3)
        u2 = torch.cat([x2, u2], 1)

        u1 = self.u1(u2)
        u1 = torch.cat([x1, u1], 1)     
        return u1, embedding