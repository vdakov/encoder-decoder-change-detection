import csv
import sys




sys.path.insert(1, 'siamese_fcn')
sys.path.insert(1, 'datasets')
sys.path.insert(1, 'evaluation')
sys.path.insert(1, 'results')
sys.path.insert(1, 'visualization')
sys.path.insert(1, 'util')


import os
import torch
from torch import nn
from hiucd_dataset_loader import HIUCD_Dataset
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from cscd_dataset_loader import CSCD_Dataset
from data_examination import examine_subset
from hrscd_dataset_loader import HRSCD_Dataset
from levir_dataset_loader import LEVIR_Dataset
from metrics import evaluate_categories, evaluate_net_predictions
from plots import aggregate_category_histograms, category_histograms, compare_number_of_buildings, create_figures
from siamunet_conc import SiamUnet_conc
from siamunet_diff import SiamUnet_diff
from tables import create_categorical_tables, create_tables, load_categorical_metrics, load_metrics, store_mean_difference
from train_test import train
from unet import Unet 
from late_siam_net import SiamLateFusion
import argparse
from aggregate_training_results import plot_loss
import gc
import json




def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment_name", type = str, default = "Default")
    parser.add_argument("--epochs", type = int, default=1)
    parser.add_argument("--fp_modifier", type = int, default=10)
    parser.add_argument("--batch_size", type = int, default=32)
    parser.add_argument("--dir", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--dataset_name", type = str, default = os.path.join("..", "..", "data", "LEVIR-CD - Toy"))
    parser.add_argument("--restore_prev", type = bool, default = False)
    parser.add_argument("--generate_plots", type = bool, default = False)

    return parser.parse_args()

fusions = ["Early", "Middle-Conc", "Middle-Diff", "Late"]


def run_experiment(experiment_name, dataset_name, datasets, dataset_loaders, criterion, epochs, restore_prev=False, generate_plots=False):
    net, net_name = None, None 
    experiment_path = os.path.join('experiment_results', experiment_name)
    os.makedirs(experiment_path, exist_ok=True)

    train_set, val_set, test_set = datasets
    
    train_set_loader, val_set_loader, test_set_loader = dataset_loaders
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    aggregate_categorical = []
    colors = {"Early": 'blue', "Middle-Conc": 'orange', "Middle-Diff": 'lime', "Late": 'red'}
    
    for fusion in fusions:
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
        categorical_metrics = {}

        if os.path.exists(f'{model_path}.pth') and restore_prev == True:
            state_dict = torch.load(f'{model_path}.pth')
            net.load_state_dict(state_dict)
            
            training_metrics, validation_metrics, test_metrics = load_metrics(model_path)
            
            if not generate_plots:
                categorical_metrics = load_categorical_metrics(model_path)
                aggregate_categorical.append(categorical_metrics)
        else: 
            os.makedirs(model_path, exist_ok=True)

            training_metrics, validation_metrics = train(net, train_set, train_set_loader, val_set, criterion, device, n_epochs= epochs, save=True, save_dir = f'{model_path}.pth', skip_val = False, early_stopping = True)
            test_metrics = evaluate_net_predictions(net, criterion, test_set)
            create_tables(training_metrics, validation_metrics, test_metrics, net_name, os.path.join(model_path, 'tables'))

        
        create_figures(training_metrics, validation_metrics, test_metrics, net_name, os.path.join(model_path, 'figures'))
        examine_subset(net, net_name, test_dataset, 3, device, os.path.join(model_path, 'figures'))
        
        if dataset_name == "CSCD" and generate_plots:
            categorical_metrics = evaluate_categories(net, dataset_name, test_set, ["large_change_uniform", "large_change_non_uniform", "small_change_non_uniform", "small_change_uniform"], os.path.join(model_path, 'figures'))
            aggregate_categorical.append(categorical_metrics)
            create_categorical_tables(categorical_metrics, net_name, os.path.join(model_path, 'tables'))
            category_histograms(net_name, 'Categorical Metrics', categorical_metrics, os.path.join(model_path, 'figures'))
        if dataset_name == "HRSCD"and generate_plots:
            categorical_metrics = evaluate_categories(net, dataset_name, test_set, ["No information", "Artificial surfaces", "Agricultural areas", 
                                                                                    "Forests", "Wetlands", "Water"], os.path.join(model_path, 'figures'))
            aggregate_categorical.append(categorical_metrics)
            create_categorical_tables(categorical_metrics, net_name, os.path.join(model_path, 'tables'))
            category_histograms(net_name, 'Categorical Metrics', categorical_metrics, os.path.join(model_path, 'figures'))
        if dataset_name == "HIUCD"and generate_plots:
            categorical_metrics = evaluate_categories(net, dataset_name, test_set, ["Unlabeled", "Water", "Grass", "Building", "Green house", 
                                                                                    "Road", "Bridge", "Others", "Bare land", "Woodland"], os.path.join(model_path, 'figures'))
            aggregate_categorical.append(categorical_metrics)
            create_categorical_tables(categorical_metrics, net_name, os.path.join(model_path, 'tables'))
            category_histograms(net_name, 'Categorical Metrics', categorical_metrics, os.path.join(model_path, 'figures'))
        if dataset_name == "LEVIR" and generate_plots:
            categorical_metrics = evaluate_categories(net, dataset_name, test_set, [], os.path.join(model_path, 'figures'))
            aggregate_categorical.append(categorical_metrics)
            create_categorical_tables(categorical_metrics, net_name, os.path.join(model_path, 'tables'))
        
    
    if generate_plots:
        with open(os.path.join(experiment_path, 'aggregate_categorical.csv'), 'w', newline='') as csv_file:
            fieldnames = aggregate_categorical[0].keys()
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(aggregate_categorical)
    
    plot_loss(experiment_name, fusions, colors)
    store_mean_difference(aggregate_categorical, experiment_path)
    
    if dataset_name == "CSCD" or dataset_name == "HRSCD" or dataset_name == "HIUCD":
        aggregate_category_histograms(dataset_name, 'Aggregate Categorical', aggregate_categorical, os.path.join(experiment_path))
    compare_number_of_buildings(dataset_name, '# Predicted Buildings', aggregate_categorical, os.path.join(experiment_path))
    

def get_dataset(dataset_name, dirname, mode, FP_MODIFIER):
    if dataset_name == 'LEVIR':
        return LEVIR_Dataset(dirname, mode, FP_MODIFIER)
    elif dataset_name == 'HRSCD':
        return HRSCD_Dataset(dirname, mode, FP_MODIFIER)
    elif dataset_name == 'CSCD':
        return CSCD_Dataset(dirname, mode, FP_MODIFIER)
    elif dataset_name == 'HIUCD':
        return HIUCD_Dataset(dirname, mode, FP_MODIFIER)
    elif dataset_name == 'LEVIR':
        return LEVIR_Dataset(dirname, mode, FP_MODIFIER)
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
    


    train_dataset = get_dataset(dataset_name, directory, "train", FP_MODIFIER)
    val_dataset = get_dataset(dataset_name, directory, "val", FP_MODIFIER)
    test_dataset = get_dataset(dataset_name, directory, "test", FP_MODIFIER)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights = torch.FloatTensor(train_dataset.weights).to(device)
    criterion = nn.NLLLoss(weight=weights)

    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)
    val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = True, num_workers = 4)

    run_experiment(experiment_name, dataset_name, [train_dataset, val_dataset, test_dataset], 
                   [train_loader, val_loader, test_loader], criterion, n_epochs,
                   restore_prev=restore_prev, generate_plots=generate_plots)

    




     




