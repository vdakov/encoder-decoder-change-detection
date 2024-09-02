import torch
import numpy as np
from tqdm import tqdm as tqdm

def reshape_for_torch(img):
    """Transpose image for PyTorch coordinates."""
    out = img.transpose((2, 0, 1))
    return torch.from_numpy(out)
