import matplotlib.pyplot as plt
import numpy as np


def compare_distributions_num_changes(dataset_name, ground_truth, predictions, colors, save_path):
    plt.hist(ground_truth, color= (0, 0, 0, 0.3))
    plt.title(dataset_name)
    for k in predictions.keys():
        plt.hist(predictions[k], color=colors[k], alpha=0.3, label=k)
    plt.xlabel('#Num Changes')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.show() 
    
def compare_distributions_sizes(dataset_name, ground_truth, predictions, colors, save_path):
    plt.hist(ground_truth, color= (0, 0, 0, 0.3))
    plt.title(dataset_name)
    for k in predictions.keys():
        plt.hist(predictions[k], color=colors[k], alpha=0.3, label=k)
    plt.xlabel('OChange Size')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.show() 
    
def compare_distributions_spread(dataset_name, ground_truth, predictions, colors, save_path):
    plt.hist(ground_truth, color= (0, 0, 0, 0.3))
    plt.title(dataset_name)
    for k in predictions.keys():
        plt.hist(predictions[k], color=colors[k], alpha=0.3, label=k)
    plt.xlabel('Connectedness')
    plt.ylabel('Frequency')
    plt.savefig(save_path)
    plt.show() 
    
    

