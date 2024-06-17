import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import os 
from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']




def plot_loss(experiment_name, fusions, colors):
    path = os.path.join('experiment_results', experiment_name)
    plt.figure(figsize=(7, 4))
    font_properties = {'family': 'serif', 'serif': ['Times New Roman'], 'weight': 'normal', 'size': 12}

    plt.xlabel('Epochs', fontdict=font_properties)
    plt.ylabel('Loss', fontdict=font_properties)

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
    
    plt.title(' Loss', weight='bold', fontname='Times New Roman')
    plt.legend()
    plt.savefig(os.path.join(path, 'aggregated_loss.png'))
    plt.tight_layout()
    plt.show()
    
    
# experiment_name = 'CSCD-FIRST-EXPERIMENT'
# fusions = ["Early", "Middle-Conc", "Middle-Diff", "Late"]
# colors = {"Early": 'blue', "Middle-Conc": 'orange', "Middle-Diff": 'lime', "Late": 'red'}

# plot_loss(experiment_name, fusions, colors)