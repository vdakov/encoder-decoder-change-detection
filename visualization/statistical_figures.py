import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from distance import calculate_connectedness
from num_objects import calculate_num_objects
from sizes import calculate_sizes

'''
The final evaluation is based on a bunch of hypothesis tests, more specifically non-parametric ones about stochastic dominance. 
Here I have functionality about plotting probability function, either histograms approximation of the PDFs, KDE approximation or 
cumilative (empirical) density functions. Along with all of that there are some utility functions within them.

All of the distributions are subsequently output into their own folders. 
There are both pairwise comparisons and aggrgated one within all other ones. 

'''


def calculate_bins(data):
    '''Calculate the optimal number of bins for the provided data array
    using the Freedman-Diaconis rule'''
    
    data = np.array(data)

    q25, q75 = np.percentile(data, [25, 75])
    bin_width = 2 * (q75 - q25) * (len(data) ** (-1/3))
    
    bins = int((data.max() - data.min()) / bin_width)
    return bins if bins > 0 else 10

def filter_zero_bins(values):
    '''
    Method that filters out all zero-values in the provided data array. 
    As the patches-cutting method produced lots of negative samples, this was useful for visual comparison purposes in some experiments.
    '''
    values = np.array(values)
    non_zero_indices = values > 0
    return values[non_zero_indices]

def compare_distributions(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros = False):
    '''Histogram comparison based on histograms. The function gets all of the data (fusions' predictions, ground truth) and outputs them 
    as a histogram with an optimal number of bins based on a pre-defined rule. It gives the KDEs of all of them as orientation as well.
    The reasoons Seaborn instead of pyplot is purely because of convenience. 
    '''
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
    
def compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros = False):
    ''' Function that displays kernel density estimates (via Parzen) of all of the data we have. Initially the distribution comparison started as so, but 
    It is functionally the same as the histogram one functionally otherwise. 
    '''
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
    
    
def compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path, x_label, y_label, filter_zeros=False):
    ''' A function that outputs a visualization of all of the empirative distribution functions (approximation of the CDFs) for our datasets. 
    It does so both pair-wise and aggregated for all. Visualizations are in the appropriate folder. The plotting is done via sorting the points against 
    probabilites, a standard EDF method.
    '''
    if filter_zeros:
        ground_truth_data = filter_zero_bins(ground_truth)
    else:
        ground_truth_data = ground_truth

    pairwise_path = os.path.join(os.path.dirname(save_path), 'pairwise')
    os.makedirs(pairwise_path, exist_ok=True)
    seen = set()
    for k1 in predictions.keys():
        for k2 in predictions.keys():
            if (k1, k2) in seen or k1 == k2: 
                continue
            seen.add((k1, k2))

            sorted_pred_k1 = np.sort(predictions[k1])
            sorted_pred_k2 = np.sort(predictions[k2])
            cdf_pred_k1 = np.arange(1, len(sorted_pred_k1) + 1) / len(sorted_pred_k1)
            cdf_pred_k2 = np.arange(1, len(sorted_pred_k2) + 1) / len(sorted_pred_k2)
            
            plt.figure(figsize=(15, 6))
            plt.plot(sorted_pred_k1, cdf_pred_k1, color=colors[k1], label=f'{k1}, $\mu$={np.mean(predictions[k1]):.2f}, $\sigma^2$={np.var(predictions[k1]):.2f}', alpha=0.5)
            plt.plot(sorted_pred_k2, cdf_pred_k2, color=colors[k2], label=f'{k2}, $\mu$={np.mean(predictions[k2]):.2f}, $\sigma^2$={np.var(predictions[k1]):.2f}', alpha=0.5)
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0., fontsize='small')
            plt.tight_layout()
            plt.savefig(os.path.join(pairwise_path, f'{k1}-{k2}_{x_label}_cdf.png'))
            plt.show()

    plt.figure(figsize=(15, 6))
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
    
    
def aggregate_distribution_histograms(dataset_name, ground_truth, predictions_dict, colors, save_path):
    '''The function that outputs all plots for the statistical comparisons dones in the study. This means 
    Histograms, EDFs, and KDEs. It aggregates all needed predictions and then outputs the figures in the appropriate folders of the experiment.'''
    gt_num_changes = calculate_num_objects(ground_truth)
    gt_spread , _ = calculate_connectedness(ground_truth, {}, scale_for_img=False)
    gt_sizes , _ = calculate_sizes(ground_truth, {}, scale_for_img=False)
    
    predictions_num_changes = {}   
    for key in predictions_dict.keys():
        predictions_num_changes[key] = calculate_num_objects(predictions_dict[key])
    
    _, predictions_spread = calculate_connectedness([], predictions_dict, scale_for_img=False)

    _, predictions_sizes = calculate_sizes([], predictions_dict, scale_for_img=False)
    
    os.makedirs(os.path.join(save_path, 'kdes'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'cdfs'), exist_ok=True)

        
    compare_distributions_num_changes(dataset_name, gt_num_changes, predictions_num_changes, colors, save_path, f'{dataset_name}aggregated_dist_num_changes.png')
    compare_distributions_spread(dataset_name, gt_spread, predictions_spread, colors, save_path, f'{dataset_name}-aggregated_dist_spread.png') 
    compare_distributions_sizes(dataset_name, gt_sizes, predictions_sizes, colors, save_path, f'{dataset_name}-aggregated_dist_sizes.png')  
    
    
    

    
'''
--------------------------------------
COMMENT FOR ALL THREE FUNCTIONS BELOW:
--------------------------------------
They realize all the visualizations for all of our three metrics. It is where all arguments, such as filtering out 
zero-bins or plot names and axises are changed. Those are the methods in the perfomances of statistical visualizations 
in 'experiment.py' 
'''
    
def compare_distributions_num_changes(dataset_name, ground_truth, predictions, colors, save_path, img_name):
    save_path_kde = os.path.join(save_path, 'kdes', img_name)
    save_path_histograms = os.path.join(save_path, 'histograms', img_name)
    save_path_cdf = os.path.join(save_path, 'cdfs', img_name)
    
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path_histograms, '#Num Changes', 'Frequency')
    compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path_kde, '#Num Changes', 'Frequency')
    compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path_cdf, '#Num Changes', 'Probability')


def compare_distributions_sizes(dataset_name, ground_truth, predictions, colors, save_path, img_name):
    save_path_kde = os.path.join(save_path, 'kdes', img_name)
    save_path_histograms = os.path.join(save_path, 'histograms', img_name)
    save_path_cdf = os.path.join(save_path, 'cdfs', img_name)
    
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path_histograms, 'Change Size', 'Frequency')
    compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path_kde, 'Change Size', 'Frequency')
    compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path_cdf, 'Change Size', 'Probability')

def compare_distributions_spread(dataset_name, ground_truth, predictions, colors, save_path, img_name):
    save_path_kde = os.path.join(save_path, 'kdes', img_name)
    save_path_histograms = os.path.join(save_path, 'histograms', img_name)
    save_path_cdf = os.path.join(save_path, 'cdfs', img_name)
    
    compare_distributions(dataset_name, ground_truth, predictions, colors, save_path_histograms, 'Connectedness', 'Frequency')
    compare_distributions_kde(dataset_name, ground_truth, predictions, colors, save_path_kde, 'Connectedness', 'Frequency')
    compare_distributions_cdf(dataset_name, ground_truth, predictions, colors, save_path_cdf, 'Connectedness', 'Probability')
