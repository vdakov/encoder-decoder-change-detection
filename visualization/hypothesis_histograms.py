import os
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

def compare_distributions(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros = False):
    bins = calculate_bins(ground_truth)
    plt.figure(figsize=(15, 6))
    
    if filter_zeros:
        ground_truth_data = filter_zero_bins(ground_truth)
    else:
        ground_truth_data = ground_truth
    
    sns.histplot(ground_truth_data, bins=calculate_bins(ground_truth_data), kde=False, 
                 color='black', label=f'Ground Truth, $\mu$={np.mean(ground_truth_data):.2f}, $\sigma^2$={np.var(ground_truth_data):.2f}', alpha=0.5)
    sns.histplot(ground_truth_data, bins=calculate_bins(ground_truth_data), kde=True, 
                 color='black', label=None, linestyle='--', linewidth=2)
    
    for k in predictions.keys():
        if filter_zeros:
            prediction_data = filter_zero_bins(predictions[k])
        else:
            prediction_data = predictions[k]
        
        sns.histplot(prediction_data, bins=calculate_bins(prediction_data), kde=False, 
                     color=colors[k], label=f'{k}, $\mu$={np.mean(prediction_data):.2f}, $\sigma^2$={np.var(prediction_data):.2f}'
                     , alpha=0.5)
        
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
    
def compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros = True):
    plt.figure(figsize=(15, 6))
    
    if filter_zeros:
        ground_truth_data = filter_zero_bins(ground_truth)
    else:
        ground_truth_data = ground_truth
    
    sns.kdeplot(ground_truth_data, 
                 color='black', label=f'Ground Truth, $\mu$={np.mean(ground_truth_data):.2f}, $\sigma^2$={np.var(ground_truth_data):.2f}', alpha=0.5)

    
    for k in predictions.keys():
        if filter_zeros:
            prediction_data = filter_zero_bins(predictions[k])
        else:
            prediction_data = predictions[k]
        
        sns.kdeplot(prediction_data,  
                     color=colors[k], label=f'{k}, $\mu$={np.mean(prediction_data):.2f}, $\sigma^2$={np.var(prediction_data):.2f}'
                     , alpha=0.5)

    
    plt.title(dataset_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    xmin = 0
    xmax = max(np.percentile(ground_truth_data, 95), np.percentile(prediction_data, 95))
    plt.xlim(xmin, xmax)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    plt.tight_layout()
    plt.savefig(save_path.split(".")[0] + "_kde.png")
    plt.show()
    
    
def compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros=True):
    plt.figure(figsize=(15, 6))
    
    if filter_zeros:
        ground_truth_data = filter_zero_bins(ground_truth)
    else:
        ground_truth_data = ground_truth

    sorted_gt = np.sort(ground_truth_data)
    cdf_gt = np.arange(1, len(sorted_gt) + 1) / len(sorted_gt)
    plt.plot(sorted_gt, cdf_gt, color='black', label=f'Ground Truth, $\mu$={np.mean(ground_truth_data):.2f}, $\sigma^2$={np.var(ground_truth_data):.2f}', alpha=0.5)

    for k in predictions.keys():
        if filter_zeros:
            prediction_data = filter_zero_bins(predictions[k])
        else:
            prediction_data = predictions[k]

        sorted_pred = np.sort(prediction_data)
        cdf_pred = np.arange(1, len(sorted_pred) + 1) / len(sorted_pred)
        plt.plot(sorted_pred, cdf_pred, color=colors[k], label=f'{k}, $\mu$={np.mean(prediction_data):.2f}, $\sigma^2$={np.var(prediction_data):.2f}', alpha=0.5)

    plt.title(dataset_name)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    
    xmin = 0
    xmax = max(np.percentile(ground_truth_data, 95), np.percentile(prediction_data, 95))
    plt.xlim(xmin, xmax)
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
    plt.tight_layout()
    plt.savefig(save_path.split(".")[0] + "_cdf.png")
    plt.show()
    
def compare_distributions_num_changes(dataset_name, ground_truth, predictions, colors, save_path):
    save_path_kde = os.path.join(save_path, 'kdes')
    save_path_histograms = os.path.join(save_path, 'histograms')
    save_path_cdf = os.path.join(save_path, 'cdfs')
    
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path_histograms, '#Num Changes', 'Frequency')
    compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path_kde, '#Num Changes', 'Frequency')
    compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path_cdf, '#Num Changes', 'Frequency')

def compare_distributions_sizes(dataset_name, ground_truth, predictions, colors, save_path):
    save_path_kde = os.path.join(save_path, 'kdes')
    save_path_histograms = os.path.join(save_path, 'histograms')
    save_path_cdf = os.path.join(save_path, 'cdfs')
    
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path_histograms, 'Change Size', 'Frequency')
    compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path_kde, 'Change Size', 'Frequency')
    compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path_cdf, 'Change Size', 'Frequency')

def compare_distributions_spread(dataset_name, ground_truth, predictions, colors, save_path):
    save_path_kde = os.path.join(save_path, 'kdes')
    save_path_histograms = os.path.join(save_path, 'histograms')
    save_path_cdf = os.path.join(save_path, 'cdfs')
    
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path_histograms, 'Connectedness', 'Frequency')
    compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path_kde, 'Connectedness', 'Frequency')
    compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path_cdf, 'Connectedness', 'Frequency')
