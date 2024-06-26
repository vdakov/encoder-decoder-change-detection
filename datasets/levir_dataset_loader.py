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



class LEVIR_Dataset(Dataset):
    
    def __init__(self, dirname, set_name, FP_MODIFIER):
        self.imgs_1 = {}
        self.imgs_2 = {}
        self.change_maps = {}
        self.items = []
        self.num_changes = {}
        
        self.dirname = dirname
        self.set_name = set_name
        self.names = []
        n_pix = 0
        true_pix = 0

        for name in os.listdir(os.path.join(dirname,set_name, "A")):
            img_name = set_name + "-" + name
            self.names.append(img_name)
            a = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name,"A", name)))
            b = reshape_for_torch(cv2.imread(os.path.join(dirname,set_name, "B", name)))
            label = cv2.imread(os.path.join(dirname, set_name, "label", name), cv2.IMREAD_GRAYSCALE)
            label = (label - np.min(label)) / (np.ptp(label)) if np.ptp(label) != 0 else np.zeros_like(label)

            self.imgs_1[img_name] = a
            self.imgs_2[img_name] = b
            self.change_maps[img_name] = label.astype(np.uint8)
            self.num_changes[img_name] = len(cv2.findContours(label.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])

            s = label.shape
            n_pix += np.prod(s)
            true_pix += label.sum()
            
            s = self.imgs_1[img_name].shape

            self.weights = [ FP_MODIFIER * 2 * true_pix / n_pix, 2 * (n_pix - true_pix) / n_pix]

    def get_img(self, im_name):
        return self.imgs_1[im_name], self.imgs_2[im_name], self.change_maps[im_name], 'placeholder', self.num_changes[im_name]

    def __len__(self):
        return len(list(self.imgs_1.keys()))

    def __getitem__(self, idx):
        im_name = list(self.imgs_1.keys())[idx]
        I1 = self.imgs_1[im_name]
        I2 = self.imgs_2[im_name]
        label = self.change_maps[im_name]
        num_changes = self.num_changes[im_name]
  
        
        sample = {'I1': I1, 'I2': I2, 'label': label, "num_changes" : num_changes}
        

        return sample
    

