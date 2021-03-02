# -*- coding: utf-8 -*-
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from network.network_unet import UNet
from network.network_densenet import DenseNet
from network.network_resnet import ResNet

class Solver():
    def __init__(self, args):
        # 参数设置
        if args.device:
            self.device = torch.device(args.device)
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.save_path = args.save_path

        self.patch_size = args.patch_size

        self.lr = args.lr

        self.net = ResNet().to(self.device)

        # L1loss()
        self.criterion = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)

        self.losses_train = []
        self.losses_set = []

    def set_input(self, input_):
        # print(input_[0].shape)
        h, w = input_[0].shape[-2:]
        self.input_ = input_[0].reshape((-1, 1, h, w)).to(self.device)
        self.target_ = input_[1].reshape((-1, 1, h, w)).to(self.device)

        # self.input_ = input_[0].to(self.device)
        # self.target_ = input_[1].to(self.device)

    def forward(self):
        self.pred_ = self.net(self.input_)

    def backward(self):
        self.loss = self.criterion(self.pred_, self.target_) + 0.2 * self.sf(self.pred_, self.target_)
        self.loss.backward()

    def sf(self, x, y):
        def sobel(inp):
            kernel_x = [[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]]
            kernel_x = torch.cuda.FloatTensor(kernel_x).unsqueeze(0).unsqueeze(0)
            weight_x = nn.Parameter(data=kernel_x, requires_grad=False)
            Gx = F.conv2d(inp, weight_x, padding=1)

            kernel_y = [[ 1,  2,  1],
                        [ 0,  0,  0],
                        [-1, -2, -1]]
            kernel_y = torch.cuda.FloatTensor(kernel_y).unsqueeze(0).unsqueeze(0)
            weight_y = nn.Parameter(data=kernel_y, requires_grad=False)
            Gy = F.conv2d(inp, weight_y, padding=1)

            G = torch.sqrt(Gx**2 + Gy**2)
            G = (G >= 0.5).float()
            return G

        return self.criterion(sobel(x), sobel(y))

    def train(self, epoch_end):
        self.forward()
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

        # loss操作
        self.losses_set.append(self.loss.item())
        if epoch_end:
            self.losses_train.append(np.mean(self.losses_set))
            self.losses_set = []
        return self.loss


    def evalute(self):
        self.forward()
        input_ = self.input_.cpu().detach().numpy()
        pred_ = self.pred_.cpu().detach().numpy()
        target_ = self.target_.cpu().detach().numpy()
        return np.mean(np.abs(pred_ - target_))*3700, (input_, target_, pred_)

    def loss_data(self):
        return self.losses_train

    def save_model(self, epoch):
        # 创建文件夹
        root = os.path.join(self.save_path, 'saved_model')
        if not os.path.exists(root):
            os.makedirs(root)
            print('Create path : {}'.format(root))
        f = os.path.join(self.save_path, 'saved_model', '{}epoch.ckpt'.format(epoch))
        torch.save(self.net.state_dict(), f)

    def load_model(self, iter_):
        f = os.path.join(self.save_path, 'saved_model','{}epoch.ckpt'.format(iter_))
        self.net.load_state_dict(torch.load(f))