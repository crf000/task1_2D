# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
import os
import numpy as np
import time


# 打印训练状态
def print_current_state(epoch, num_epochs, iter_, iter_sum, start_time, loss):
    print("EPOCH [{}/{}], TIME [{:.1f}s], ITER [{}/{}], LOSS: {:.8f}"
          .format(epoch, num_epochs, time.time() - start_time, iter_ + 1, iter_sum, loss.item()))


# 打印loss图
def plot_current_loss(losses_train, save_path):
    x = range(len(losses_train))
    plt.plot(x, losses_train, label='losses_train', color='g', linewidth=0.5)
    plt.legend()
    plt.savefig(os.path.join(save_path, 'loss.png'))
    plt.close()


# 打印转换图  横向四张
def plot_images(image_tuple, epoch_, slice, save_path):
    # 创建文件夹
    root = os.path.join(save_path, 'epoch_'+str(epoch_))
    if not os.path.exists(root):
        os.makedirs(root)
        print('Create path : {}'.format(root))

    MR, CT, pCT = image_tuple
    diff = np.abs(CT - pCT)

    MR, CT, pCT, diff = MR.squeeze(), CT.squeeze(), pCT.squeeze(), diff.squeeze()
    f, ax = plt.subplots(1, 4, figsize=(30, 10))
    ax[0].imshow(MR, cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
    ax[0].set_title('MR', fontsize=30)
    ax[0].axis('off')

    ax[1].imshow(CT, cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
    ax[1].set_title('target CT', fontsize=30)
    ax[1].axis('off')

    ax[2].imshow(pCT, cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
    ax[2].set_title('predicted CT', fontsize=30)
    ax[2].axis('off')

    ax[3].imshow(diff, cmap=plt.cm.gray, vmin=0.0, vmax=1.0)
    ax[3].set_title('above difference', fontsize=30)
    ax[3].axis('off')

    # 保存
    plt.savefig(os.path.join(root, 'img%d.png' % slice))
    # 关闭
    plt.close()