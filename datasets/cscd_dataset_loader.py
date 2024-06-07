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
import csv

FP_MODIFIER = 1


class CSCD_Dataset(Dataset):

    def __init__(self, dirname, set_name, stride, patch_side=0):
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.situations = {}

        self.stride = stride
        self.dirname = dirname
        self.set_name = set_name
        self.names = []
        self.items = []
        n_pix = 0
        true_pix = 0

        situation_dict = {}
        with open(os.path.join(dirname, set_name, "situation_labels.csv"), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                im_name = row['im_name']
                situation = row['situation']
                situation_dict[im_name] = situation

        names = set(os.listdir(os.path.join(dirname,set_name, "A"))) & set(os.listdir(os.path.join(dirname,set_name, "B"))) & set(os.listdir(os.path.join(dirname,set_name, "label"))) & situation_dict.keys()


        for name in names:
            img_name = set_name + "-" + name
            self.names.append(img_name)
            a = reshape_for_torch(cv2.imread(os.path.join(dirname, set_name,"A", name)))
            b = reshape_for_torch(cv2.imread(os.path.join(dirname, set_name, "B", name)))
            label = cv2.imread(os.path.join(dirname, set_name, "label", name), cv2.IMREAD_GRAYSCALE)
            situation = situation_dict[name]


            self.imgs_1[img_name] = a
            self.imgs_2[img_name] = b
            self.change_maps[img_name] = label
            self.situations[img_name] = situation

            s = label.shape
            n_pix += np.prod(s)
            true_pix += label.sum()

            self.items.append((a, b, label, situation))

        self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name], self.situations[im_name]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        I1, I2, label, situation = self.items[idx]

        sample = {'I1': I1, 'I2': I2, 'label': label, "situation": situation}


        return sample


