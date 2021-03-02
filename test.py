# -*- coding: utf-8 -*-
import numpy as np
import argparse
import torch

from solver import Solver
from dataLoader import get_loader
import visualizer

def TestArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='')
    parser.add_argument('--data_path', type=str, default=r'/mnt/match')
    parser.add_argument('--save_path', type=str, default=r'/mnt/task1_2D/result')
    parser.add_argument('--save_iters', type=int, default=20)
    parser.add_argument('--test_iters', type=int, default=40, help='20 40 60 80')

    parser.add_argument('--batch_size', type=int, default=1, help='input batch size')


    ###### 不用管，用不到 ########
    parser.add_argument('--patch_n', type=int, default=8)
    parser.add_argument('--patch_size', type=tuple, default=(16, 128, 128))
    parser.add_argument('--drop_background', type=float, default=0.1)

    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=1e-4)
    ###### 不用管，用不到 ########

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = TestArgs()
    test_data_loader = get_loader(args, 'test')
    model = Solver(args)

    print('iter_:' + str(args.test_iters))
    with torch.no_grad():
        MAE = []
        model.load_model(args.test_iters)
        for idx, data in enumerate(test_data_loader):
            model.set_input(data)
            MAE_, image_tuple = model.evalute()
            MAE.append(MAE_)
            visualizer.plot_images(image_tuple, args.test_iters, idx, args.save_path)

        print('epoch{}-MAE:{:.2f}+-{:.2f}'.format(args.test_iters, np.mean(MAE), np.std(MAE)))
