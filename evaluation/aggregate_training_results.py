import numpy as np 
import csv 
import matplotlib.pyplot as plt 
import os 
from matplotlib import rcParams

###########################
# Functions meant to aggregate dictionaries of evaluation metrics. As we are interested in looking at all architecture 
# performances on the same plot, we want to be apble to plot them as such.
###########################

rcParams["font.family"] = "Times New Roman"
rcParams['font.size'] = 24  # You can change this to the desired font size
rcParams['axes.titlesize'] = 24  # Title font size
rcParams['axes.labelsize'] = 24  # Axis label font size
rcParams['xtick.labelsize'] = 24  # X tick label font size
rcParams['ytick.labelsize'] = 24  # Y tick label font size





def plot_loss(experiment_name, fusions, colors):
    '''
    Produces an aggregated loss plot from existing CSV files. I know the structure is weird, but
    it avoids cluttering the code with other things while in the time constraints of the project. 
    '''
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
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)
    plt.savefig(os.path.join(path, 'aggregated_loss.png'))
    plt.show()
    
