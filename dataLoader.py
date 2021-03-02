# -*- coding: utf-8 -*-
import os
from glob import glob
import numpy as np
from torch.utils.data import Dataset, DataLoader


class dataset(Dataset):
    def __init__(self, data_type, args):
        self.data_type = data_type
        self.data_path = args.data_path
        self.patch_size = args.patch_size
        self.patch_n = args.patch_n
        self.drop_background = args.drop_background

        # print(os.path.join(self.data_path, self.data_type, 'case*_MR1.npy'))
        MR1 = glob(os.path.join(self.data_path, self.data_type, 'case*_MR1.npy'))
        MR2 = glob(os.path.join(self.data_path, self.data_type, 'case*_MR2.npy'))
        CT = glob(os.path.join(self.data_path, self.data_type, 'case*_CT.npy'))

        self.input_ = [np.load(f) for f in MR1]
        self.input_ = np.vstack(self.input_)#[:, np.newaxis]
        self.target_ = [np.load(f) for f in CT]
        self.target_ = np.vstack(self.target_)#[:, np.newaxis]
        print(self.input_.shape)

    def __len__(self):
        return len(self.input_)

    def __getitem__(self, idx):
        input_img = self.input_[idx]
        target_img = self.target_[idx]
        # return input_img, target_img

        # patch
        if self.data_type == 'test':
            return input_img, target_img

        elif self.data_type == 'train':
            input_patches, target_patches = get_patch(input_img,
                                                      target_img,
                                                      self.patch_n,
                                                      self.patch_size,
                                                      self.drop_background)
            return input_patches, target_patches   



def get_patch(full_input_img, full_target_img, patch_n, patch_size, drop_background):
    assert full_input_img.shape == full_target_img.shape
    patch_input_imgs = []
    patch_target_imgs = []
    h, w = full_input_img.shape
    new_h, new_w = patch_size
    n = 0

    while n < patch_n:
        h_head = np.random.randint(0, h - new_h)
        w_head = np.random.randint(0, w - new_w)
        patch_input_img = full_input_img[h_head:h_head + new_h, w_head:w_head + new_w]
        patch_target_img = full_target_img[h_head:h_head + new_h, w_head:w_head + new_w]

        # 去掉背景
        # if np.mean(patch_target_img) < drop_background:
        #     continue
        # else:
        n += 1
        patch_input_imgs.append(patch_input_img)
        patch_target_imgs.append(patch_target_img)

    return np.array(patch_input_imgs), np.array(patch_target_imgs)


def get_loader(args, data_type):
    dataset_ = dataset(data_type, args)
    data_loader = DataLoader(dataset=dataset_,
                             batch_size=args.batch_size,
                             shuffle=True if data_type == 'train' else False,
                             num_workers=6)
    return data_loader


if __name__ == '__main__':
    pass
    # a = get_loader('E:\\科研任务1\\processed_dataset\\match', 'train', 8, 4, (4, 32, 32), 0.5)
    # print(a.dataset.input_[15])
    # print(a.dataset.target_[15])
