import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision.transforms as tr
import os
import numpy as np
from tqdm import tqdm as tqdm
import pandas 
import cv2
import os
from math import floor, ceil, sqrt, exp


def reshape_for_torch(img):
    """Transpose image for PyTorch coordinates."""
    out = img.transpose((2, 0, 1))
    return torch.from_numpy(out)