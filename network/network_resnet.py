# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os



class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        block_n = 3
        layers = [InConv(1, 32)]
        for _ in range(0, block_n):
            layers.append(ResidualBlock(32, 32))
        layers.append(OutConv(32, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.basic = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        out = self.basic(x)
        residual = x
        # print(out.shape)
        # print(residual.shape)
        out += residual
        return out


class InConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InConv, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        return self.net(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.net = nn.Sequential(nn.BatchNorm2d(in_channels),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels, out_channels, kernel_size=1))

    def forward(self, x):
        return self.clip_by_tensor(self.net(x), 0.0, 1.0)

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


if __name__ == '__main__':
    pass
    # batch size, dim, D, H, W
    # a = torch.rand(1, 1, 56, 512, 512)
    # net = ResNet_3D()
    # print(net)
    # b = net(a)
    # print(b.shape)
