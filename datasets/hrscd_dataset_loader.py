import sys

sys.path.insert(1, '../siamese_fcn')
sys.path.insert(1, '../datasets')
sys.path.insert(1, '../evaluation')
sys.path.insert(1, '../results')
sys.path.insert(1, '../visualization')
sys.path.insert(1, '..')
sys.path.insert(1, '../util')


from torch.utils.data import Dataset
import os
from preprocess_util import reshape_for_torch
import numpy as np
from tqdm import tqdm as tqdm
import cv2
import os
from math import ceil

FP_MODIFIER = 1


class HRSCD_Dataset(Dataset):

    def __init__(self, dirname, set_name, stride, patch_side):
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.land_cover_1 = {}
        # self.land_cover_2 = {}

        self.stride = stride
        self.dirname = dirname
        self.n_patches_per_image = {}
        self.n_patches = 0
        self.patch_coords = []
        self.patch_side = 1000
        self.set_name = set_name
        self.names = []
        n_pix = 0
        true_pix = 0

        l = len(os.listdir(os.path.join(dirname,set_name, "A")))


        for name in os.listdir(os.path.join(dirname,set_name, "A")):

            img_name = set_name + "-" + name
            self.names.append(img_name)
            a = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name,"A", name)))
            b = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name, "B", name)))
            label = cv2.imread(os.path.join(dirname, set_name, "labels", name), cv2.IMREAD_GRAYSCALE)
            land_cover_1 =  cv2.imread(os.path.join(dirname, set_name, "labels_land_cover_A", name))
            # land_cover_2 =  cv2.imread(os.path.join(dirname, set_name, "labels_land_cover_B", name), cv2.IMREAD_GRAYSCALE)

            self.imgs_1[img_name] = a
            self.imgs_2[img_name] = b
            self.change_maps[img_name] = label
            self.land_cover_1[img_name] = land_cover_1



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

                    current_patch_coords = (img_name,
                                    [self.stride*i, self.stride*i + self.patch_side, self.stride*j, self.stride*j + self.patch_side],
                                    [self.stride*(i + 1), self.stride*(j + 1)])

                    self.patch_coords.append(current_patch_coords)

        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name], self.land_cover_1[im_name]

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
        landcover_labels_1 = self.land_cover_1[im_name][limits[0]:limits[1], limits[2]:limits[3]]
        # landcover_labels_2 = self.land_cover_2[im_name][limits[0]:limits[1], limits[2]:limits[3]]


        sample = {'I1': I1, 'I2': I2, 'label': label, "landcover": landcover_labels_1}


        return sample


