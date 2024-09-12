import csv
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib import font_manager, rcParams

from statistical_figures import calculate_bins


font_path = 'Times New Roman.ttf'
try:
    prop = font_manager.FontProperties(fname=font_path)
    font_manager.fontManager.addfont(font_path)
    rcParams['font.family'] = prop.get_name()
except Exception as e:
    print(f"Warning: Could not load custom font. Using default font. Error: {e}")



def create_loss_accuracy_figures(train_metrics, val_metrics, test_metrics, model_name, save_path = None):
    '''The function that creates the big figure with the Loss, Accuracy and Precision-Recall Plots. It 
    evaluates all of them for their training, validation and test results at the end.
    '''
    _, axs = plt.subplots(1, 3, figsize=(17, 5))
    
    print(train_metrics, val_metrics, test_metrics)

    train_loss = extract_metric(train_metrics, 'net_loss')
    val_loss = extract_metric(val_metrics, 'net_loss')
    test_loss = test_metrics['net_loss']

    train_accuracy = extract_metric(train_metrics, 'net_accuracy')
    val_accuracy = extract_metric(val_metrics, 'net_accuracy')
    test_accuracy = test_metrics['net_accuracy']

    train_precision = extract_metric(train_metrics, 'precision')
    val_precision = extract_metric(val_metrics, 'precision')
    test_precision = test_metrics['precision']

    train_recall = extract_metric(train_metrics, 'recall')
    val_recall = extract_metric(val_metrics, 'recall')
    test_recall = test_metrics['recall']


    train_f1 = [2 * p * r /  (p + r+ 1e-10) for p, r in zip(train_precision, train_recall)]
    val_f1 = [2 * p * r / (p + r+ 1e-10)  for p, r in zip(val_precision, val_recall)]
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall + 1e-10)



    plt.suptitle(model_name)

    axs[0].plot(train_loss, label='Train', c='blue')

    axs[0].plot(val_loss, label='Validation', c='orange')
    axs[0].axhline(y=test_loss, color='red', linestyle='--', label='Test')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')

    axs[1].plot(train_accuracy, label='Train', c='blue')
    axs[1].plot(val_accuracy, label='Validation', c='orange')
    axs[1].axhline(y=test_accuracy, color='red', linestyle='--', label='Test')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')

    axs[2].plot(train_f1, label='Train F1')
    axs[2].plot(val_f1, label='Validation F1')
    axs[2].axhline(y=test_precision, color='red', linestyle='--', label='Test Precision')
    axs[2].axhline(y=test_recall, color='green', linestyle='--', label='Test Recall')
    axs[2].axhline(y=test_f1, color='blue', linestyle='--', label='Test F1')
    axs[2].set_title('Precision, Recall, and F1 Curve')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Score')

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'loss-accuracy-precision.png'))
        


def extract_metric(metric_list, key):
    '''Just a helper for extracting only a given metric because the other syntax was annoying me.'''
    return [item[key] for item in metric_list]


def load_loss_data(file_path):
    """Helper function to load loss data from a CSV file."""
    results = []
    try:
        with open(file_path, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                results.append(float(row['net_loss']))
    except FileNotFoundError:
        print(f"Warning: File {file_path} not found.")
    except KeyError:
        print(f"Warning: 'net_loss' column not found in {file_path}.")
    return results

def plot_aggregated_loss(experiment_name, fusions, colors):
    '''
    Produces an aggregated loss plot from existing CSV files. The idea is to see 
    how all of the test architectures converge. 
    '''
    path = os.path.join('experiment_results', experiment_name)
    plt.figure(figsize=(15, 5))
    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    max_loss_value = 0
    
    for f in fusions:
        dir = f + "-" + experiment_name 
        train_file_path = os.path.join(path, dir, 'tables', 'train_metrics.csv')
        val_file_path = os.path.join(path, dir, 'tables', 'val_metrics.csv')
        
        training_results = load_loss_data(train_file_path)
        validation_results = load_loss_data(val_file_path)
        
        if training_results:
            max_loss_value = max(max_loss_value, max(training_results))
            plt.plot(np.arange(len(training_results)), training_results, label=f, color=colors[f], linestyle='-')
        
        # Plot validation results
        if validation_results:
            max_loss_value = max(max_loss_value, max(validation_results))
            plt.plot(np.arange(len(validation_results)), validation_results, label=f'{f}-Val.', color=colors[f], linestyle='dashed')
    
    if max_loss_value > 0:
        plt.ylim([0, max_loss_value])
        
    plt.tight_layout()
    plt.legend()
    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(os.path.join(path, 'aggregated_loss.png'))
    plt.show()
    



    


def plot_comparison_histogram(dataset_name, gt, predictions, save_path):
    num_bins = max(calculate_bins(gt), calculate_bins(predictions))
    plt.hist(gt, bins = num_bins, color = 'black', alpha=0.5)
    plt.hist(predictions, bins = num_bins, color = 'blue', alpha=0.5)
    plt.savefig(save_path)
    plt.show()
    
    

    
def plot_aggregated_loss(experiment_name, fusions, colors):
    '''
    Produces an aggregated loss plot from existing CSV files. 
    '''
    path = os.path.join('experiment_results', experiment_name)
    plt.figure(figsize=(15, 5))
    

    plt.xlabel('Epochs')
    plt.ylabel('Loss')


    for f in fusions:
        dir = f + "-" + experiment_name 
        
        with open(os.path.join(path, dir, 'tables', 'train_metrics.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            training_results = []
            for row in reader:
                training_results.append(float(row['net_loss']))
                
            plt.ylim([0, max(training_results) if len(training_results) > 0 else 0])
            plt.plot(np.arange(len(training_results)), training_results, label=f, color=colors[f], linestyle='-')
        
        with open(os.path.join(path, dir, 'tables', 'val_metrics.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            validation_results = []
            for row in reader:
                validation_results.append(float(row['net_loss']))
            plt.ylim([0, max(validation_results) if len(validation_results) > 0 else 0])
            plt.plot(np.arange(len(validation_results)), validation_results, label=f'{f}-Val.', color=colors[f], linestyle='dashed')
    
    plt.tight_layout()
    plt.legend()
    # plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(os.path.join(path, 'aggregated_loss.png'))
    plt.show()
    

    
    

        

