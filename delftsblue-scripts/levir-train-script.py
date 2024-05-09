import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import evaluate_net_predictions
from tables import create_tables
from visualize import create_figures
from tqdm import tqdm as tqdm
import sys
sys.path.insert(1, 'siamese_fcn')
sys.path.insert(1, 'datasets')
from unet import Unet
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from fresunet import FresUNet
from levir_dataset_loader import LEVIR_Dataset
import time
from train_test import train

FP_MODIFIER = 1 # Tuning parameter, use 1 if unsure
BATCH_SIZE = 32
PATCH_SIDE = 96
N_EPOCHS = 50

dirname = "..\\data\\LEVIR-CD"

train_dataset = LEVIR_Dataset(dirname, "train", PATCH_SIDE)
weights = torch.FloatTensor(train_dataset.weights).cuda()
train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

test_dataset = LEVIR_Dataset(dirname, "test", PATCH_SIDE)
test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

val_dataset = LEVIR_Dataset(dirname, "val", PATCH_SIDE)
val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

net, net_name = FresUNet(2*3, 2), 'FresUNet'
net.cuda()
criterion = nn.NLLLoss(weight=weights)



t_start = time.time()
save_dir = f'{net_name}-{time.time()}.pth.tar'
training_metrics = train(net, net_name, train_dataset, train_loader, val_dataset, criterion, n_epochs=10, save=True, save_dir = save_dir)
t_end = time.time()
print('Elapsed time:')
print(t_end - t_start)

test_metrics = evaluate_net_predictions(net, criterion, test_dataset)
create_figures(training_metrics, test_metrics, net_name)
create_tables(training_metrics, test_metrics, net_name)