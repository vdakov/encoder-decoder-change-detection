import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from metrics import evaluate_net_predictions
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from train_test import train
from unet import Unet 
from late_siam_net import SiamLateFusion

def run_experiment(experiment_name, datasets, dataset_loaders, fusion, criterion):
    net, net_name = None, None 
    patch_size = 100
    n_epochs = 10

    dataset_name, train_set, val_set, test_set = datasets
    train_set_loader, val_set_loader, test_set_loader = dataset_loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    if fusion is "Early":
        net, net_name = Unet(3, 2), f'Early-{experiment_name}'
    elif fusion is "Middle-Conc": 
        net, net_name = SiamUnet_conc(3, 2), f'Middle-Conc-{experiment_name}'
    elif fusion is "Middle-Diff":
        net, net_name = SiamUnet_diff(3, 2), f'Middle-Conc-{experiment_name}'
    elif fusion is "Late": 
        net, net_name = SiamLateFusion(3, 2), f'Late-{experiment_name}'

    net.to(device)

    save_path = os.path.join('trained_models', net_name, f'{net_name}.pth')
    os.makedirs(os.path.join('trained_models', net_name), exist_ok=True)
    training_metrics = None 

    if os.path.exists(os.path.join('trained_models', net_name, f'{net_name}.pth')):
        net = torch.load(save_path)
        net.to(device)
        show_evaluations()
        
        
        
    else: 
        training_metrics = train(net, net_name, train_set, train_set_loader, val_set, criterion, patch_size, device, n_epochs=n_epochs, save=True, save_dir = save_path)

    test_metrics = evaluate_net_predictions(net, criterion, test_set, patch_size)
    show_evaluations(net, net_name, training_metrics, test_metrics)







     
def show_evaluations():
    dataset_name = None
    if dataset_name == 'LEVIR-CD':
        show_levir_plots()
    if dataset_name == 'HRSCD': 
        show_hrscd_plots()
    if dataset_name == 'HiUCD':
        show_hiucd_plots()
    if dataset_name == 'CSCD': 
        show_cscd_plots()



def show_levir_plots():
    pass 

def show_hiucd_plots():
    pass 

def show_hrscd_plots():
    pass 

def show_cscd_plots():
    pass 

