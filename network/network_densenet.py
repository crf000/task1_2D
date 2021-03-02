""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os


class DenseNet(nn.Module):
    def __init__(self, growth_rate=16, blocks_n=4, bn_size=4):
        super(DenseNet, self).__init__()
        self.net = nn.Sequential(
            in_conv(1, growth_rate),
            dense_block(growth_rate, blocks_n, bn_size),
            out_conv(growth_rate * (blocks_n + 1), 1)
        )

    def forward(self, x):
        return self.net(x)


class dense_block(nn.Module):
    def __init__(self, growth_rate, blocks_n, bn_size):
        super(dense_block, self).__init__()
        layers = []
        for i in range(blocks_n):
            layers.append(conv_block(growth_rate * (i + 1), growth_rate, bn_size))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class conv_block(nn.Module):
    def __init__(self, in_channels, growth_rate, bn_size):
        super(conv_block, self).__init__()
        self.net = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, bn_size * growth_rate, 1, 1, 0),
            nn.BatchNorm2d(bn_size * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size * growth_rate, growth_rate, 3, 1, 1))

    def forward(self, x):
        x_ = self.net(x)
        return torch.cat((x, x_), dim=1)


class in_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(in_conv, self).__init__()
        self.net = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, 1, 1))

    def forward(self, x):
        return self.net(x)


class out_conv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(out_conv, self).__init__()
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
