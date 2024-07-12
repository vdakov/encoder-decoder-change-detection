import matplotlib.pyplot as plt
import numpy as np


colors = {
    "FC-EF": (0, 0, 1), 
    "FC-Siam-Conc.": (1, 0.62, 0), 
    "FC-Siam-Diff.": (0, 1, 0), 
    "FC-LF": (1, 0, 0),
}

def compare_distributions_num_changes(dataset_name, ground_truth, predictions):
        

    plt.hist(ground_truth, color= (0, 0, 0, 0.3))
    plt.title(dataset_name)
    for k in predictions.keys():
        plt.hist(predictions[k], color=colors[k], alpha=0.3, label=k)
    plt.xlabel('#Num Changes')
    plt.ylabel('Frequency')
    plt.show() 
    
def compare_distributions_sizes(dataset_name, ground_truth, predictions):
    plt.hist(ground_truth, color= (0, 0, 0, 0.3))
    plt.title(dataset_name)
    for k in predictions.keys():
        plt.hist(predictions[k], color=colors[k], alpha=0.3, label=k)
    plt.xlabel('OChange Size')
    plt.ylabel('Frequency')
    plt.show() 
    
def compare_distributions_spread(dataset_name, ground_truth, predictions):
    plt.hist(ground_truth, color= (0, 0, 0, 0.3))
    plt.title(dataset_name)
    for k in predictions.keys():
        plt.hist(predictions[k], color=colors[k], alpha=0.3, label=k)
    plt.xlabel('Connectedness')
    plt.ylabel('Frequency')
    plt.show() 
    
    


ground_truth = np.random.normal(170, 10, 1000)
predictions = {
    "FC-EF": np.random.normal(120, 10, 1000), 
    "FC-Siam-Conc.": np.random.normal(250, 10, 1000), 
    "FC-Siam-Diff.": np.random.normal(70, 10, 1000), 
    "FC-LF": np.random.normal(160, 10, 1000)
}

compare_distributions_num_changes('LEVIR', ground_truth, predictions)