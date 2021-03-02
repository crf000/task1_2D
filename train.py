# -*- coding: utf-8 -*-
import time
from solver import Solver
from dataLoader import get_loader
import visualizer
import os
import argparse

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'


def TrainArgs():
    parser = argparse.ArgumentParser()

    parser.add_argument('--device', type=str, default='')
    # E:/科研任务1/processed_dataset/match  /mnt/match
    parser.add_argument('--data_path', type=str, default=r'/mnt/match')
    # ./result  /mnt/task1_2D/result
    parser.add_argument('--save_path', type=str, default=r'/mnt/task1_2D/result')

    parser.add_argument('--batch_size', type=int, default=16, help='input batch size')
    parser.add_argument('--epochs', type=int, default=61)
    parser.add_argument('--lr', type=float, default=1e-4)

    parser.add_argument('--save_iters', type=int, default=20)

    parser.add_argument('--patch_n', type=int, default=24)
    parser.add_argument('--patch_size', type=tuple, default=(128, 128))
    parser.add_argument('--drop_background', type=float, default=0.1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = TrainArgs()
    # get_loader(data_path, data_type, batch_size, patch_n, patch_size, drop_background):
    train_data_loader = get_loader(args, 'train')
    model = Solver(args)

    start_time = time.time()
    iter_sum = len(train_data_loader)
    for epoch in range(args.epochs):
        # 训练
        for iter_, data in enumerate(train_data_loader):
            model.set_input(data)
            loss = model.train(epoch_end=(iter_ + 1 == iter_sum))
            visualizer.print_current_state(epoch, args.epochs, iter_, iter_sum, start_time, loss)

        losses_train = model.loss_data()
        visualizer.plot_current_loss(losses_train, args.save_path)

        # 保存模型
        if epoch % args.save_iters == 0 and epoch != 0:
            model.save_model(epoch)
