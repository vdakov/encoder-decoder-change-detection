import os
import numpy as np 
from scipy import stats

from distance import calculate_distances
from hypothesis_histograms import compare_distributions_num_changes, compare_distributions_sizes, compare_distributions_spread
from num_objects import calculate_num_objects
from plots import plot_comparison_histogram
from sizes import calculate_sizes


def perform_kruskal_wallis(dataset_name, predictions, output_file, p_val = 0.05):
    '''
    Compute the Kruskal-Wallis H-test for all of our predictions in terms of size, number and spread per fusion architecture.
    The Kruskal-Wallis H-test tests the null hypothesis that the population median of all of the groups are equal. It is a non-parametric version of ANOVA. It
    indicates that *at least one* sample stochastically dominates the other. 
    The test works on 2 or more independent samples, which may have different sizes. Note that rejecting the null hypothesis does not indicate which of the groups differs. 
    Post hoc comparisons between groups are required to determine which groups are different.
    '''
    rejected = False
    
    predictions_list = []
    for fusion in predictions.keys():
        predictions_list.append(predictions[fusion])
    
    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        h_statistic, p = stats.kruskal(predictions_list)
            
        if p < p_val:
            rejected = True
        
        if not rejected:
            file.write(f'Accept the null hypothesis with an H={h_statistic}!\n')
        else:
            file.write(f'Reject the null hypothesis with an H={h_statistic}!\n')
            
    return rejected 

            
            
def perform_kruskal_wallis_per_metric(dataset_name, predictions_dict, save_path, p_val=0.05):
    

    predictions_num_changes = {}   
    for key in predictions_dict.keys():
        predictions_num_changes[key] = calculate_num_objects(dataset_name, predictions_dict[key])
    _, predictions_spread = calculate_distances(dataset_name, [], predictions_dict)
    _, predictions_sizes = calculate_sizes(dataset_name, [], predictions_dict)
    

    
    num_changes_first_order_sgd = perform_kruskal_wallis(dataset_name, predictions_num_changes, os.path.join(save_path, 'kruskal_wallis_num_changes.txt'), p_val)
    spread_first_order_sgd = perform_kruskal_wallis(dataset_name, predictions_spread, os.path.join(save_path, 'kruskal_wallis_spread.txt'), p_val)
    sizes_first_order_sgd = perform_kruskal_wallis(dataset_name, predictions_sizes, os.path.join(save_path, 'kruskal_wallis_size.txt'), p_val)
    
    if num_changes_first_order_sgd:
        perform_post_hoc_comparison_first_order_sgd(dataset_name, 'num_changes', predictions_num_changes, os.path.join(save_path, 'mann_whitney_num_changes.txt'), p_val)
    if spread_first_order_sgd:
        perform_post_hoc_comparison_first_order_sgd(dataset_name, 'spread', predictions_spread, os.path.join(save_path, 'mann_whitney_spread.txt'), p_val)
    if sizes_first_order_sgd: 
        perform_post_hoc_comparison_first_order_sgd(dataset_name, 'sizes', predictions_sizes, os.path.join(save_path, 'kruskal_wallis_num_changes.txt'), p_val)
    
def perform_post_hoc_comparison_first_order_sgd(dataset_name, metric, predictions_dict, output_file, p_val=0.05):
    
    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        for k1 in predictions_dict.keys():
            for k2 in predictions_dict.keys():
                if k1 == k2: 
                    continue

                u_statistic, p = stats.mannwhitneyu(predictions_dict[k1], predictions_dict[k2], alternative='greater')
                if p > p_val:
                    file.write(f'{k1} first-order-stochastically dominates {k2} for {metric} with U={u_statistic}')
                    
            

    
    
    
def aggregate_distribution_histograms(dataset_name, ground_truth, predictions_dict, colors, save_path):
    gt_num_changes = calculate_num_objects(dataset_name, ground_truth)
    gt_spread , _ = calculate_distances(dataset_name, ground_truth, {})
    gt_sizes , _ = calculate_sizes(dataset_name, ground_truth, {})
    
    predictions_num_changes = {}   
    for key in predictions_dict.keys():
        predictions_num_changes[key] = calculate_num_objects(dataset_name, predictions_dict[key])
    
    _, predictions_spread = calculate_distances(dataset_name, [], predictions_dict)

    _, predictions_sizes = calculate_sizes(dataset_name, [], predictions_dict)
    
    os.makedirs(os.path.join(save_path, 'kdes'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'cdfs'), exist_ok=True)

        
    compare_distributions_num_changes(dataset_name, gt_num_changes, predictions_num_changes, colors, save_path, f'{dataset_name}aggregated_dist_num_changes.png')
    compare_distributions_spread(dataset_name, gt_spread, predictions_spread, colors, save_path, f'{dataset_name}-aggregated_dist_spread.png') 
    compare_distributions_sizes(dataset_name, gt_sizes, predictions_sizes, colors, save_path, f'{dataset_name}-aggregated_dist_sizes.png')  
    
    
    


        

