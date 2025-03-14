import sys

import torch

from num_objects import get_number_of_objects

sys.path.insert(1, '../siamese_fcn')
sys.path.insert(1, '../datasets')
sys.path.insert(1, '../evaluation')
sys.path.insert(1, '../results')
sys.path.insert(1, '../visualization')
sys.path.insert(1, '..')
sys.path.insert(1, '../preprocessing')



from torch.utils.data import Dataset
import os
from reshape import reshape_for_torch
import numpy as np
from tqdm import tqdm as tqdm
import cv2
import os
from math import ceil

# def get_dataset(dataset_name, dirname, mode, FP_MODIFIER, patch_side=96, transform=None):
#     if dataset_name == 'LEVIR':
#         return LEVIR_Dataset(dirname, mode, FP_MODIFIER, patch_side=patch_side, transform=transform)
#     elif dataset_name == 'HRSCD':
#         return HRSCD_Dataset(dirname, mode, FP_MODIFIER, patch_side=patch_side, transform=transform)
#     elif dataset_name == 'CSCD':
#         return CSCD_Dataset(dirname, mode, FP_MODIFIER, patch_side=patch_side, transform=transform)
#     elif dataset_name == 'HIUCD':
#         return HIUCD_Dataset(dirname, mode, FP_MODIFIER, patch_side=patch_side, transform=transform)
#     else:
#         raise ValueError("Unknown dataset name")


class HIUCD_Dataset(Dataset):

    def __init__(self, dirname, set_name, FP_MODIFIER, patch_side = 96, transform=None):
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.land_covers = {}
        self.num_changes = {}

        self.dirname = dirname
        self.set_name = set_name
        self.names = []
        n_pix = 0
        true_pix = 0
        
        self.transform = transform
        self.patch_side = patch_side
        self.stride = int(patch_side/2) - 1
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        
        
        
        names_imgs = set(os.listdir(os.path.join(dirname,set_name, "A"))) & set(os.listdir(os.path.join(dirname,set_name, "B"))) & set(os.listdir(os.path.join(dirname,set_name, "label"))) 
        self.names = [] 

        for name in names_imgs:

            img_name = set_name + "-" + name
            self.names.append(img_name)
            a = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name,"A", name)))
            b = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name, "B", name)))
            label = cv2.imread(os.path.join(dirname, set_name, "label", name))
            
            
            landcover_label = label[:, :, 2].astype(np.uint8)
            label = np.where(label[:, :, 0] < 2, 0, 255).astype(np.uint8)
            label = (label - np.min(label)) / (np.ptp(label)) if np.ptp(label) != 0 else np.zeros_like(label)
            

            self.imgs_1[img_name] = a
            self.imgs_2[img_name] = b
            self.change_maps[img_name] = label.astype(np.uint8) 
            self.land_covers[img_name] = landcover_label
            self.num_changes[img_name] = get_number_of_objects(label)

            s = label.shape
            n_pix += np.prod(s)
            true_pix += label.sum()
            
            s = self.imgs_1[img_name].shape
            n1 = ceil((s[1] - self.patch_side + 1) / self.stride)
            n2 = ceil((s[2] - self.patch_side + 1) / self.stride)
            n_patches_i = n1 * n2
            self.n_patches_per_image[img_name] = n_patches_i
            self.n_patches += n_patches_i
            
            for i in range(n1):
                for j in range(n2):
                    # coordinates in (x1, x2, y1, y2)
                    current_patch_coords = (img_name, 
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)], landcover_label)
                    self.patch_coords.append(current_patch_coords)

        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name], self.land_covers[im_name], self.num_changes[im_name]

    def __len__(self):
        return self.n_patches

    def __getitem__(self, idx):
        # im_name = list(self.imgs_1.keys())[idx]

        # I1 = self.imgs_1[im_name]
        # I2 = self.imgs_2[im_name]
        # label = self.change_maps[im_name]
        # landcover_labels = self.land_covers[im_name]
        # num_changes = self.num_changes[im_name]
        
        # return {'I1': I1, 'I2': I2, 'label': label, "landcover": landcover_labels, "num_changes" : num_changes}
        
        current_patch_coords = self.patch_coords[idx]
        im_name = current_patch_coords[0]
        limits = current_patch_coords[1]
        centre = current_patch_coords[2]
        
        
        I1 = self.imgs_1[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        I2 = self.imgs_2[im_name][:, limits[0]:limits[1], limits[2]:limits[3]]
        
        label = self.change_maps[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        num_changes = get_number_of_objects(label)
        label = torch.from_numpy(1*np.array(label)).float()
        
        landcover = current_patch_coords[3]
        
        sample = {'I1': I1, 'I2': I2, 'label': label, "landcover": landcover, "num_changes" : num_changes}
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    


