# import matplotlib.pyplot as plt
# import numpy as np


# def compare_distributions_num_changes(dataset_name, ground_truth, predictions, colors, save_path):
#     bins = calculate_bins(ground_truth)
#     plt.hist(ground_truth, bins = bins, color= (0, 0, 0, 0.7))
#     plt.title(dataset_name)
#     for k in predictions.keys():
#         bins = calculate_bins(predictions[k])
#         plt.hist(predictions[k], bins = bins, color=colors[k], alpha=0.7, label=k)
#     plt.xlabel('#Changes')
#     plt.ylabel('Frequency')
#     plt.savefig(save_path)
#     plt.show() 
    
# def compare_distributions_sizes(dataset_name, ground_truth, predictions, colors, save_path):
#     bins = calculate_bins(ground_truth)
#     plt.hist(ground_truth, bins = bins, color= (0, 0, 0, 0.7))
#     plt.title(dataset_name)
#     for k in predictions.keys():
#         bins = calculate_bins(predictions[k])
#         plt.hist(predictions[k], bins = bins, color=colors[k], alpha=0.7, label=k)
#     plt.xlabel('Change Area')
#     plt.ylabel('Frequency')
#     plt.savefig(save_path)
#     plt.show()
    
     
    
# def compare_distributions_spread(dataset_name, ground_truth, predictions, colors, save_path):
#     bins = calculate_bins(ground_truth)
#     plt.hist(ground_truth, bins = bins, color= (0, 0, 0, 0.7))
#     plt.title(dataset_name)
#     for k in predictions.keys():
#         bins = calculate_bins(predictions[k])
#         plt.hist(predictions[k], bins = bins, color=colors[k], alpha=0.7, label=k)
#     plt.xlabel('Connectedness')
#     plt.ylabel('Frequency')
#     plt.savefig(save_path)
#     plt.show() 
    

# def calculate_bins(data):
#     # Calculate the optimal number of bins using the Freedman-Diaconis rule
#     data = np.array(data)

#     q25, q75 = np.percentile(data, [25, 75])
#     bin_width = 2 * (q75 - q25) * (len(data) ** (-1/3))
    
   
#     bins = int((data.max() - data.min()) / bin_width)
#     return bins if bins > 0 else 10
    
    

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def calculate_bins(data):
    # Calculate the optimal number of bins using the Freedman-Diaconis rule
    data = np.array(data)

    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * (len(data) ** (-1/3))
    
   
    bins = int((data.max() - data.min()) / bin_width)
    return bins if bins > 0 else 10

def filter_zero_bins(values):
    values = np.array(values)
    non_zero_indices = values > 0
    return values[non_zero_indices]

def compare_distributions(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros = True):
    bins = calculate_bins(ground_truth)
    plt.figure(figsize=(15, 6))
    
    if filter_zeros:
        ground_truth_data = filter_zero_bins(ground_truth)
    else:
        ground_truth_data = ground_truth
    
    sns.histplot(ground_truth_data, bins=calculate_bins(ground_truth_data), kde=False, 
                 color='black', label=f'Ground Truth, $\mu$={np.mean(ground_truth_data):.2f}, $\sigma^2$={np.var(ground_truth_data):.2f}', 
                 edgecolor='black', alpha=0.5)
    sns.histplot(ground_truth_data, bins=calculate_bins(ground_truth_data), kde=True, 
                 color='black', label=None, linestyle='--', linewidth=2)
    
    for k in predictions.keys():
        if filter_zeros:
            prediction_data = filter_zero_bins(predictions[k])
        else:
            prediction_data = predictions[k]
        
        sns.histplot(prediction_data, bins=calculate_bins(prediction_data), kde=False, 
                     color=colors[k], label=f'{k}, $\mu$={np.mean(prediction_data):.2f}, $\sigma^2$={np.var(prediction_data):.2f}', 
                     edgecolor='black', alpha=0.5)
        
        sns.histplot(prediction_data, bins=calculate_bins(prediction_data), kde=True, 
                     color=colors[k], label=None, linestyle='--', linewidth=2)
    
    plt.title(dataset_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    xmin = 0
    xmax = max(np.percentile(ground_truth_data, 95), np.percentile(prediction_data, 95))
    plt.xlim(xmin, xmax)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    
def compare_distributions_num_changes(dataset_name, ground_truth, predictions, colors, save_path):
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path, '#Num Changes', 'Frequency')

def compare_distributions_sizes(dataset_name, ground_truth, predictions, colors, save_path):
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path, 'Change Size', 'Frequency')

def compare_distributions_spread(dataset_name, ground_truth, predictions, colors, save_path):
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path, 'Connectedness', 'Frequency')
