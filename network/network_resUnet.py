import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os

class ResUNet(nn.Module):
    def __init__(self, inp_channels=1, base_channels=16, resBlock_n=3):
        super(ResUNet, self).__init__()
        self.base_channels = base_channels

        self.resNet = []
        for _ in range(0, resBlock_n):
            self.resNet.append(ResidualBlock(self.base_channels, self.base_channels))
        self.resNet = nn.Sequential(*self.resNet)

        self.inc = DoubleConv(inp_channels, self.base_channels)
        self.down1 = Down(self.base_channels, self.base_channels * 2)
        self.down2 = Down(self.base_channels * 2, self.base_channels * 4)
        self.down3 = Down(self.base_channels * 4, self.base_channels * 8)
        self.down4 = Down(self.base_channels * 8, self.base_channels * 8)
        self.up1 = Up(self.base_channels * 16, self.base_channels * 4)
        self.up2 = Up(self.base_channels * 8, self.base_channels * 2)
        self.up3 = Up(self.base_channels * 4, self.base_channels)
        self.up4 = Up(self.base_channels * 2, self.base_channels)
        self.outc = OutConv(self.base_channels, 1)

    def forward(self, x):
        x1 = self.inc(x)
        res = self.resNet(x1)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, res)
        return self.outc(x)

class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.clip_by_tensor(self.conv(x), 0.0, 1.0)

    def clip_by_tensor(self, t, t_min, t_max):
        """
        clip_by_tensor
        :param t: tensor
        :param t_min: min
        :param t_max: max
        :return: cliped tensor
        """
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.basic = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        out = self.basic(x)
        return torch.add(out, x)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels , in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


if __name__ == '__main__':
    from torchstat import stat
    import torchvision.models as models

    net = ResUNet()
    # print(net)
    input = torch.randn(1, 1, 512, 512)
    # print(net(input))
    stat(net, input)

