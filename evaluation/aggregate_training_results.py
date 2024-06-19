import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import os 
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']
rcParams['font.size'] = 24  # You can change this to the desired font size
rcParams['font.weight'] = 'bold'
rcParams['axes.titlesize'] = 24  # Title font size
rcParams['axes.titleweight'] = 'bold'  # Title font weight
rcParams['axes.labelsize'] = 24  # Axis label font size
rcParams['axes.labelweight'] = 'bold'  # Axis label font weight
rcParams['xtick.labelsize'] = 24  # X tick label font size
rcParams['ytick.labelsize'] = 24  # Y tick label font size





def plot_loss(experiment_name, fusions, colors):
    path = os.path.join('experiment_results', experiment_name)
    plt.figure(figsize=(8, 5))
    

    plt.xlabel('Epochs')
    plt.ylabel('Loss')

    for f in fusions:
        dir = f + "-" + experiment_name 
        
        with open(os.path.join(path, dir, 'tables', 'train_metrics.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            training_results = []
            for row in reader:
                training_results.append(float(row['net_loss']))
            plt.plot(np.arange(len(training_results)), training_results, label=f, color=colors[f], linestyle='-')
        
        with open(os.path.join(path, dir, 'tables', 'val_metrics.csv'), newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            validation_results = []
            for row in reader:
                validation_results.append(float(row['net_loss']))
            plt.plot(np.arange(len(validation_results)), validation_results, label=f'{f}-Val.', color=colors[f], linestyle='dashed')
    
    # plt.title('Loss', weight='bold')
    plt.savefig(os.path.join(path, 'aggregated_loss.png'))
    plt.tight_layout()
    plt.show()
    
