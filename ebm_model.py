import torch
from torch import nn
from torch.nn import functional as F
from neural_operations import Conv2D

def Lip_swish(x):
    return (x * torch.sigmoid(x))/1.1


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, downsample=False, data_init=True):
        super().__init__()

        self.conv1 = Conv2D(in_channel, out_channel, 3, padding=1, bias=True, data_init=data_init)
        self.conv2 = Conv2D(out_channel, out_channel, 3, padding=1, bias=True, data_init=data_init)
        self.skip = None

        if in_channel != out_channel or downsample:
            self.skip = nn.Sequential(Conv2D(in_channel, out_channel, 1, bias=False,data_init=data_init))

        self.downsample = downsample

    def forward(self, input):
        out = self.conv1(input)
        out = Lip_swish(out)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)
        else:
            skip = input
        out = out + skip

        if self.downsample:
            out = F.avg_pool2d(out, 2)

        out = Lip_swish(out)
        return out


class EBM(nn.Module):
    def __init__(self, nc=3, mid_channel=128, data_init=False):
        super().__init__()

        self.conv1 = Conv2D(nc, mid_channel, 3, padding=1, bias=True, data_init=data_init)

        self.blocks = nn.ModuleList([
            ResBlock(mid_channel, mid_channel, downsample=True, data_init=data_init),
            ResBlock(mid_channel, mid_channel, data_init=data_init),
            ResBlock(mid_channel, mid_channel * 2, downsample=True, data_init=data_init),
            ResBlock(mid_channel * 2, mid_channel * 2, data_init=data_init),
            ResBlock(mid_channel * 2, mid_channel * 2, downsample=True, data_init=data_init),
            ResBlock(mid_channel * 2, mid_channel * 2, data_init=data_init),
        ])
        self.linear = nn.Linear(2 * mid_channel, 1)

        self.all_conv_layers = []
        for n, layer in self.named_modules():
            if isinstance(layer, Conv2D):
                self.all_conv_layers.append(layer)

        self.sr_u = {}
        self.sr_v = {}
        self.num_power_iter = 4

    def forward(self, input):
        out = self.conv1(input)

        out = Lip_swish(out)

        for block in self.blocks:
            out = block(out)

        out = out.view(out.shape[0], out.shape[1], -1).sum(2)
        out = self.linear(out)

        return out.squeeze(1)
