import os
import numpy as np 
from scipy import stats

from distance import calculate_connectedness, calculate_distances
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
        if predictions[fusion]:  
            predictions_list.append(predictions[fusion])
            
    assert len(predictions_list) >= 2
    output_dir = os.path.dirname(output_file)
    assert os.path.exists(output_dir)
    
    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')    
        h_statistic, p = stats.kruskal(*predictions_list)
            
        if p < p_val:
            rejected = True
        
        if not rejected:
            file.write(f'Accept the null hypothesis with an H={h_statistic}!\n')
        else:
            file.write(f'Reject the null hypothesis with an H={h_statistic}!\n')
            
    return rejected 

            
def perform_kruskal_wallis_and_fosd_per_metric(dataset_name, predictions_num_changes, predictions_spread,  predictions_sizes, save_path, p_val=0.05):
    '''
    Plain helper function for both Kruskal Wallis and 
    '''
    
    num_changes_first_order_sgd = perform_kruskal_wallis(dataset_name, predictions_num_changes, os.path.join(save_path, 'kruskal_wallis_num_changes.txt'), p_val)
    spread_first_order_sgd = perform_kruskal_wallis(dataset_name, predictions_spread, os.path.join(save_path, 'kruskal_wallis_spread.txt'), p_val)
    sizes_first_order_sgd = perform_kruskal_wallis(dataset_name, predictions_sizes, os.path.join(save_path, 'kruskal_wallis_size.txt'), p_val)
    
    if num_changes_first_order_sgd:
        perform_post_hoc_comparison_first_order_sd(dataset_name, 'num_changes', predictions_num_changes, os.path.join(save_path, 'first_order_sd_mann_whitney_num_changes.txt'), p_val)
    if spread_first_order_sgd:
        perform_post_hoc_comparison_first_order_sd(dataset_name, 'spread', predictions_spread, os.path.join(save_path, 'first_order_sd_mann_whitney_spread.txt'), p_val)
    if sizes_first_order_sgd: 
        perform_post_hoc_comparison_first_order_sd(dataset_name, 'sizes', predictions_sizes, os.path.join(save_path, 'first_order_sd_mann_whitney_sizes.txt'), p_val)
    
def perform_post_hoc_comparison_first_order_sd(dataset_name, metric, predictions_dict, output_file, p_val=0.05):
    '''
    A test for first-order stochastic dominance between pairs, in case Kruskal-Wallis shows there is a need for it. 
    The Mann-Whitney test is a non-parametric measure of this as it makes no assumptions about the underlying distribution of the 
    data. 
    
    Sources:
    -https://ocw.mit.edu/courses/14-123-microeconomic-theory-iii-spring-2015/875150f8deb05a910756a02a4f9f78a5_MIT14_123S15_Chap4.pdf
    -https://en.wikipedia.org/wiki/Stochastic_dominance#
    
    '''
    
    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        keys = list(predictions_dict.keys())
        seen = set()
        for k1 in predictions_dict.keys():
            for k2 in predictions_dict.keys():
                if k1 == k2 or (k1, k2) in seen or (k2, k1) in seen:
                    continue
                seen.add((k1, k2))
                u_statistic_1, p_1 = stats.mannwhitneyu(predictions_dict[k1], predictions_dict[k2], alternative='greater')
                u_statistic_2, p_2 = stats.mannwhitneyu(predictions_dict[k2], predictions_dict[k1], alternative='greater')

                if p_1 <= p_val:
                    file.write(f'{k1} first-order-stochastically dominates {k2} with U={u_statistic_1} for {metric}!\n')
                elif p_2 <= p_val:
                    file.write(f'{k2} first-order-stochastically dominates {k1} with U={u_statistic_2} for {metric}!\n')
                else:
                    file.write(f'No first-order stochastic dominance between {k1} and {k2} for {metric}.\n')

                    
def perform_post_hoc_comparison_second_order_sd(dataset_name, metric, predictions_dict, output_file):
    '''
    In case Kruskal-Wallis finds nothing (since first order SD is sufficient for the second order), we will test for 
    SOSD via the integral theorem between all edfs. 
    
    Sources: 
        https://github.com/jamlamberti/Py4FinOpt/blob/master/02_Stochastic_Dominance.ipynb
        https://en.wikipedia.org/wiki/Stochastic_dominance#
        https://ocw.mit.edu/courses/14-123-microeconomic-theory-iii-spring-2015/875150f8deb05a910756a02a4f9f78a5_MIT14_123S15_Chap4.pdf
    '''
    edfs = {}
    for k in predictions_dict.keys():
        edfs[k] = compute_edf(predictions_dict[k])
        
    any_dominance = False 
        
    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        keys = list(predictions_dict.keys())
        seen = set()
        for k1 in predictions_dict.keys():
            for k2 in predictions_dict.keys():
                if k1 == k2 or (k1, k2) in seen or (k2, k1) in seen:
                    continue
                seen.add((k1, k2))
                points = np.sort(np.union1d(edfs[k1], edfs[k2]))
                k1_integral_area = np.cumsum([edfs[k1] * (points[i+1] - points[i]) for i in range(len(points) - 1)])
                k2_integral_area = np.cumsum([edfs[k2] * (points[i+1] - points[i]) for i in range(len(points) - 1)])
                k1_sosd_k2 = all(map(lambda x, y : x <= y, k1_integral_area, k2_integral_area))
                if k1_sosd_k2:
                    file.write(f'{k1} second-order-stochastically dominates {k2} for {metric}!\n ')
                
    
def compute_edf(data):
    
    sorted_data = np.sort(data)
    cdf_probabilites = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    return np.array([(x, y) for (x, y) in zip(sorted_data, cdf_probabilites)])


def perform_statistical_tests(dataset_name, predictions_dict, experiment_path, p_val=0.05):
    
    
    predictions_num_changes = {}   
    for key in predictions_dict.keys():
        predictions_num_changes[key] = calculate_num_objects(dataset_name, predictions_dict[key])
    _, predictions_spread = calculate_connectedness(dataset_name, [], predictions_dict, scale_for_img=False)
    _, predictions_sizes = calculate_sizes(dataset_name, [], predictions_dict, scale_for_img=False)
    
    print('Spread', predictions_spread)
    print('Sizes', predictions_sizes)
    
    perform_kruskal_wallis_and_fosd_per_metric(dataset_name, predictions_num_changes, predictions_spread, predictions_sizes, experiment_path, p_val)
    
    
    perform_post_hoc_comparison_second_order_sd(dataset_name, 'num_changes', predictions_num_changes, os.path.join(experiment_path, 'second_order_sd_num_changes.txt'))
    perform_post_hoc_comparison_second_order_sd(dataset_name, 'spread', predictions_spread, os.path.join(experiment_path, 'second_order_sd_spread.txt'))
    perform_post_hoc_comparison_second_order_sd(dataset_name, 'sizes', predictions_sizes, os.path.join(experiment_path, 'second_order_sd_sizes.txt'))
    
    
    
def aggregate_distribution_histograms(dataset_name, ground_truth, predictions_dict, colors, save_path):
    gt_num_changes = calculate_num_objects(dataset_name, ground_truth)
    gt_spread , _ = calculate_connectedness(dataset_name, ground_truth, {}, scale_for_img=False)
    gt_sizes , _ = calculate_sizes(dataset_name, ground_truth, {}, scale_for_img=False)
    
    predictions_num_changes = {}   
    for key in predictions_dict.keys():
        predictions_num_changes[key] = calculate_num_objects(dataset_name, predictions_dict[key])
    
    _, predictions_spread = calculate_connectedness(dataset_name, [], predictions_dict, scale_for_img=False)

    _, predictions_sizes = calculate_sizes(dataset_name, [], predictions_dict, scale_for_img=False)
    
    os.makedirs(os.path.join(save_path, 'kdes'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'histograms'), exist_ok=True)
    os.makedirs(os.path.join(save_path, 'cdfs'), exist_ok=True)

        
    compare_distributions_num_changes(dataset_name, gt_num_changes, predictions_num_changes, colors, save_path, f'{dataset_name}aggregated_dist_num_changes.png')
    compare_distributions_spread(dataset_name, gt_spread, predictions_spread, colors, save_path, f'{dataset_name}-aggregated_dist_spread.png') 
    compare_distributions_sizes(dataset_name, gt_sizes, predictions_sizes, colors, save_path, f'{dataset_name}-aggregated_dist_sizes.png')  
    
    
    


        

