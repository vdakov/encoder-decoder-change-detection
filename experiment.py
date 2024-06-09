import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cscd_dataset_loader import CSCD_Dataset
from data_examination import examine_subset
from hrscd_dataset_loader import HRSCD_Dataset
from levir_dataset_loader import LEVIR_Dataset
from metrics import evaluate_categories, evaluate_net_predictions
from plots import create_figures
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from tables import create_tables, load_metrics
from train_test import train
from unet import Unet 
from late_siam_net import SiamLateFusion
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type = str, default = "Default")
    parser.add_argument("--epochs", type = int, default=1)
    parser.add_argument("--fp_modifier", type = int, default=10)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--dir", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--dataset_name", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))

    return parser.parse_args()

fusions = ["Early", "Middle-Conc", "Middle-Diff", "Late"]


def run_experiment(experiment_name, dataset_name, datasets, dataset_loaders, criterion):
    net, net_name = None, None 
    experiment_path = os.path.join('experiment_results', experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    train_set, val_set, test_set = datasets
    train_set_loader, val_set_loader, test_set_loader = dataset_loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    for fusion in fusions:
        if fusion is "Early":
            net, net_name = Unet(3, 2), f'Early-{experiment_name}'
        elif fusion is "Middle-Conc": 
            net, net_name = SiamUnet_conc(3, 2), f'Middle-Conc-{experiment_name}'
        elif fusion is "Middle-Diff":
            net, net_name = SiamUnet_diff(3, 2), f'Middle-Conc-{experiment_name}'
        elif fusion is "Late": 
            net, net_name = SiamLateFusion(3, 2), f'Late-{experiment_name}'

        net.to(device)

        save_path = os.path.join(experiment_path, f'{net_name}')
        model_path = os.path.join(experiment_name, net_name)
        os.makedirs(model_path, exist_ok=True)

        

        if os.path.exists(save_path):
            net = torch.load(os.path.join(model_path, f'{net_name}.pth'))
            net.to(device)
            
            training_metrics, validation_metrics, test_metrics = load_metrics(model_path)
        else: 
            training_metrics, validation_metrics = train(net, train_set, train_set_loader, val_set, criterion, device, n_epochs= 5, save=True, save_dir = save_path, skip_val = False)
            test_metrics = evaluate_net_predictions(net, criterion, test_set)
            create_tables(training_metrics, validation_metrics, test_metrics, net_name, os.path.join(model_path, 'tables'))

    create_figures(training_metrics, validation_metrics, test_metrics, net_name, os.path.join(model_path, 'figures'))
    examine_subset(net, net_name, test_dataset, 10, device)
    
    if dataset_name is "CSCD":
        categorical_metrics = evaluate_categories(net, dataset_name, test_set, ["large_change_uniform", "large_change_non_uniform", "small_change_non_uniform", "small_change_uniform"])

def get_dataset(dataset_name, dirname, mode):
    if dataset_name == 'LEVIR':
        return LEVIR_Dataset(dirname, mode)
    elif dataset_name == 'HRSCD':
        return HRSCD_Dataset(dirname, mode)
    elif dataset_name == 'CSCD':
        return CSCD_Dataset(dirname, mode)
    else:
        raise ValueError("Unknown dataset name")
    
if __name__ == "__main__":
    args = get_args()
    dataset_name = args.dataset_name
    directory = args.dir
    batch_size = args.batch_size 

    train_dataset = get_dataset(dataset_name, directory, "train")
    val_dataset = get_dataset(dataset_name, directory, "val")
    test_dataset = get_dataset(dataset_name, directory, "test")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.FloatTensor(train_dataset.weights).to(device)
    criterion = nn.NLLLoss(weight=weights)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    run_experiment(args.experiment_name, [train_dataset, val_dataset, test_dataset], [train_loader, val_loader, test_loader], criterion, batch_size)

    




     




