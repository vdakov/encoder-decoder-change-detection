import csv
import sys






sys.path.insert(1, 'siamese_fcn')
sys.path.insert(1, 'pytorch_datasets')
sys.path.insert(1, 'evaluation')
sys.path.insert(1, 'results')
sys.path.insert(1, 'visualization')
sys.path.insert(1, 'preprocessing')


import os
import torch
from matplotlib import pyplot as plt
import numpy as np
from data_augmentation import RandomFlip, RandomRot
from losses.focal_loss import FocalLoss
import torchvision.transforms as tr
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cscd_dataset_loader import CSCD_Dataset
from data_examination import examine_subset
from levir_dataset_loader import LEVIR_Dataset
from metrics import evaluate_net_predictions, get_ground_truth, get_predictions
from plots import create_loss_accuracy_figures
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from tables import create_tables, load_metrics, store_mean_difference_per_epoch
from train_test import train
from unet import Unet 
from late_siam_net import SiamLateFusion
import argparse
from aggregate_training_results import plot_loss
import gc
import json
from statistical_tests import aggregate_distribution_histograms, perform_statistical_tests





def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type = str, default = "Default")
    parser.add_argument("--epochs", type = int, default=1)
    parser.add_argument("--fp_modifier", type = int, default=10)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--dir", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--dataset_name", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--loss", type=str, default="nll")
    parser.add_argument("--restore_prev", type = bool, default = False)
    parser.add_argument("--generate_plots", type = bool, default = False)
    parser.add_argument("--data_augmentation", type = bool, default = False)
    parser.add_argument("--patch_side", type = int, default = 96)

    return parser.parse_args()

fusions = ["Early", "Middle-Conc", "Middle-Diff", "Late"]


def run_experiment(experiment_name, dataset_name, datasets, dataset_loaders, criterion, epochs, restore_prev=False, generate_plots=False):
    '''
    A full experiment, running all four fusion architectures (so far) on a dataset with given hyperparameters. It can be set to restore the weights 
    from a previous experiment. 
    '''
    net, net_name = None, None 
    experiment_path = os.path.join('experiment_results', experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    train_set, val_set, test_set = datasets
    
    train_set_loader, val_set_loader, test_set_loader = dataset_loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    aggregate_categorical = []
    
    n = 4
    arr = plt.cm.viridis(np.linspace(0, 1, n))
    colors = {"Early":  arr[0], "Middle-Conc": arr[1], "Middle-Diff": arr[2], "Late": arr[3]}
    # colors = {"Early": 'blue', "Middle-Conc": 'orange', "Middle-Diff": 'lime', "Late": 'red'}
    predictions_dict = {}
    
    for fusion in fusions:
        print('Training ', fusion, ':')
        if fusion == "Early":
            net, net_name = Unet(6, 2), f'Early-{experiment_name}'
        elif fusion == "Middle-Conc": 
            net, net_name = SiamUnet_conc(3, 2), f'Middle-Conc-{experiment_name}'
        elif fusion == "Middle-Diff":
            net, net_name = SiamUnet_diff(3, 2), f'Middle-Diff-{experiment_name}'
        elif fusion == "Late": 
            net, net_name = SiamLateFusion(3, 2), f'Late-{experiment_name}'

        net.to(device)

        model_path = os.path.join(experiment_path, net_name)

        if os.path.exists(f'{model_path}.pth') and restore_prev == True:
            print('Restored weights!')
            state_dict = torch.load(f'{model_path}.pth')
            net.load_state_dict(state_dict)
            
            training_metrics, validation_metrics, test_metrics = load_metrics(model_path)
            

        else: 
            os.makedirs(model_path, exist_ok=True)

            training_metrics, validation_metrics = train(net, train_set, train_set_loader, val_set, val_set_loader, criterion, device, n_epochs= epochs, save=True, save_dir = f'{model_path}.pth', skip_val = False, early_stopping = True)
            test_metrics = evaluate_net_predictions(net, criterion, test_set_loader)
            create_tables(training_metrics, validation_metrics, test_metrics, os.path.join(model_path, 'tables'))

        
        test_metrics = evaluate_net_predictions(net, criterion, test_set_loader)
        create_loss_accuracy_figures(training_metrics, validation_metrics, test_metrics, net_name, os.path.join(model_path, 'figures'))
        examine_subset(net, net_name, test_dataset, 10, device, os.path.join(model_path, 'figures'))
        
        predictions = get_predictions(net, test_dataset)
        predictions_dict[fusion] = predictions
          
        

    
    plot_loss(experiment_name, fusions, colors)
    store_mean_difference_per_epoch(aggregate_categorical, experiment_path)
    ground_truth = get_ground_truth(test_dataset)
    perform_statistical_tests(dataset_name, ground_truth, predictions_dict, experiment_path, p_val=0.05)
    aggregate_distribution_histograms(dataset_name, ground_truth, predictions_dict, colors, experiment_path)
    

    

def get_dataset(dataset_name, dirname, mode, FP_MODIFIER, patch_side=96, transform=None):
    if dataset_name == 'LEVIR':
        return LEVIR_Dataset(dirname, mode, FP_MODIFIER, patch_side=patch_side, transform=transform)
    elif dataset_name == 'CSCD':
        return CSCD_Dataset(dirname, mode, FP_MODIFIER, patch_side=patch_side, transform=transform)
    else:
        raise ValueError("Unknown dataset name")
    
if __name__ == "__main__":
    args = get_args()
    experiment_name = args.experiment_name
    dataset_name = args.dataset_name
    directory = args.dir
    batch_size = args.batch_size 
    FP_MODIFIER = args.fp_modifier
    restore_prev = args.restore_prev
    n_epochs = args.epochs
    generate_plots = args.generate_plots
    loss = args.loss
    data_augmentation = args.data_augmentation
    patch_side = args.patch_side
    
    torch.manual_seed(42)
    
    if data_augmentation:
        data_transform = tr.Compose([RandomFlip(), RandomRot()])
    else:
        data_transform = None
    


    train_dataset = get_dataset(dataset_name, directory, "train", FP_MODIFIER, transform=data_transform, patch_side = patch_side)
    val_dataset = get_dataset(dataset_name, directory, "val", FP_MODIFIER, transform=data_transform, patch_side = patch_side)
    test_dataset = get_dataset(dataset_name, directory, "test", FP_MODIFIER, transform=data_transform, patch_side = patch_side)
        
  
        
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.FloatTensor(train_dataset.weights).to(device)
    
    if loss == "nll":
        criterion = nn.NLLLoss(weight=weights)
    elif loss == "focal_loss":
        criterion = FocalLoss(gamma=2, alpha=0.25)
    else: 
        criterion = nn.NLLLoss(weight=weights)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 1)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 1)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 1)

    run_experiment(experiment_name, dataset_name, [train_dataset, val_dataset, test_dataset], 
                   [train_loader, val_loader, test_loader], criterion, n_epochs,
                   restore_prev=restore_prev, generate_plots=generate_plots)

    




     




