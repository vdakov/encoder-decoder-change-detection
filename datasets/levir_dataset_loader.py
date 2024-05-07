# Imports

# PyTorch
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import torchvision.transforms as tr
import os
import numpy as np
from tqdm import tqdm as tqdm
import cv2
import os
from math import ceil


class LEVIR_Dataset(Dataset):
    
    def __init__(self, dirname, set_name, patch_side):
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.stride = 1
        

        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        self.patch_side = patch_side
        n_pix = 0
        true_pix = 0

        for name in os.listdir(os.path.join(dirname,set_name, "A")):
            img_name = set_name + "-" + name
            a = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name,"A", name)))
            b = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name, "B", name)))
            label = reshape_for_torch(cv2.imread(os.path.join(dirname, set_name, "label", name)))

            self.imgs_1[img_name] = a
            self.imgs_2[img_name] = b
            self.change_maps[img_name] = label

            s = label.shape
            n_pix += np.prod(s)
            true_pix += label.sum()
            
            # calculate the number of patches
            s = self.imgs_1[img_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[img_name] = n_patches_i
            self.n_patches += n_patches_i

            # generate path coordinates
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (img_name, 
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])
            self.patch_coords.append(current_patch_coords)
            self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        label = torch.from_numpy(1*np.array(label)).float()
        
        sample = {'I1': I1, 'I2': I2, 'label': label}
        

        return sample
    

