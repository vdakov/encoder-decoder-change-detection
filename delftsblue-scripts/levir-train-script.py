import argparse
import sys

sys.path.insert(1, '../siamese_fcn')
sys.path.insert(1, '../datasets')
sys.path.insert(1, '../evaluation')
sys.path.insert(1, '../results')
sys.path.insert(1, '../visualization')
sys.path.insert(1, '..')
sys.path.insert(1, '../util')

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from metrics import evaluate_net_predictions
from tables import create_tables
from visualize import create_figures
from tqdm import tqdm as tqdm
from preprocess_util import reshape_for_torch 
from unet import Unet
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from fresunet import FresUNet
from levir_dataset_loader import LEVIR_Dataset
import time
from train_test import train



def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type = int, default=1)
    parser.add_argument("--fp_modifier", type = int, default=1)
    parser.add_argument("--test_run", type = bool, default=False)
    parser.add_argument("--patch_side", type = int, default=96)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--dirname", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD"))
    parser.add_argument("--dataset", type = str, default = "LEVIR-CD")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    N_EPOCHS = args.epochs
    FP_MODIFIER = args.fp_modifier
    BATCH_SIZE = args.batch_size
    PATCH_SIDE = args.patch_side
    TEST_RUN = args.test_run
    DIRNAME = os.path.join("..", "..", "data", "LEVIR-CD")


if TEST_RUN:
    print("TEST")
else: 
    train_dataset = LEVIR_Dataset(DIRNAME, "train", PATCH_SIDE)
    weights = torch.FloatTensor(train_dataset.weights).cuda()
    train_loader = DataLoader(train_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    test_dataset = LEVIR_Dataset(DIRNAME, "test", PATCH_SIDE)
    test_loader = DataLoader(test_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    val_dataset = LEVIR_Dataset(DIRNAME, "val", PATCH_SIDE)
    val_loader = DataLoader(val_dataset, batch_size = BATCH_SIZE, shuffle = True, num_workers = 4)

    net, net_name = FresUNet(2*3, 2), 'FresUNet'
    net.cuda()
    criterion = nn.NLLLoss(weight=weights)



    t_start = time.time()
    save_dir = f'{net_name}-{time.time()}.pth.tar'
    training_metrics = train(net, net_name, train_dataset, train_loader, val_dataset, criterion, n_epochs=1, save=True, save_dir = save_dir)
    t_end = time.time()
    print('Elapsed time:')
    print(t_end - t_start)

    test_metrics = evaluate_net_predictions(net, criterion, test_dataset)
    create_figures(training_metrics, test_metrics, net_name)
    create_tables(training_metrics, test_metrics, net_name)