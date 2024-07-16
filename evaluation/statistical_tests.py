import os
import numpy as np 
from scipy import stats

from distance import calculate_distances
from num_objects import calculate_num_objects
from plots import plot_comparison_histogram
from sizes import calculate_sizes

def hypothesis_test_num_changes(dataset_name, ground_truth, predictions, output_file, p_val = 0.05):
    gt_mu = np.mean(ground_truth)

    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        file.write(f'Ground truth mean: {gt_mu}\n')
        
        for k in predictions.keys():
            t_test, p = stats.ttest_rel(ground_truth, predictions[k])
            
            file.write(f'{k}: {np.mean(predictions[k])}\n')
            
            if p < p_val:
                rejected = True
        
        if not rejected:
            file.write('Accept the null hypothesis about the number of changes!\n')
        else:
            file.write('Reject the null hypothesis about the number of changes!\n')
            
            
def hypothesis_test_object_size(dataset_name, ground_truth, predictions, output_file, p_val = 0.05):
    gt_mu = np.mean(ground_truth)

    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        file.write(f'Ground truth mean: {gt_mu}\n')
        
        for k in predictions.keys():
            t_test, p = stats.ttest_rel(ground_truth, predictions[k])
            
            file.write(f'{k}: {np.mean(predictions[k])}\n')
            
            if p < p_val:
                rejected = True
        
        if not rejected:
            file.write('Accept the null hypothesis about the size of changes!\n')
        else:
            file.write('Reject the null hypothesis about the size of changes!\n')
            
def hypothesis_test_object_spread(dataset_name, ground_truth, predictions, output_file, p_val = 0.05):
    gt_mu = np.mean(ground_truth)

    with open(output_file, 'w') as file:
        file.write(f'Dataset: {dataset_name}\n')
        file.write(f'Ground truth mean: {gt_mu}\n')
        
        for k in predictions.keys():
            t_test, p = stats.ttest_rel(ground_truth, predictions[k])
            
            file.write(f'{k}: {np.mean(predictions[k])}\n')
            
            if p < p_val:
                rejected = True
        
        if not rejected:
            file.write('Accept the null hypothesis about the spread of changes!\n')
        else:
            file.write('Reject the null hypothesis about the spread of changes!\n')
            
            
            
def perform_statistical_tests(dataset_name, ground_truth, predictions, save_path, p_val=0.05):
    gt_num_changes, predictions_num_changes = calculate_num_objects(dataset_name, ground_truth), calculate_num_objects(dataset_name, predictions) 
    hypothesis_test_num_changes(dataset_name, gt_num_changes, predictions_num_changes, os.path.join(save_path, 't_test_num_changes.txt'), p_val)
    
    gt_spread, predictions_spread = calculate_distances(dataset_name, ground_truth, predictions)
    hypothesis_test_object_spread(dataset_name, gt_spread, predictions_spread, os.path.join(save_path, 't_test_spread.txt'), p_val)
    gt_sizes, predictions_sizes = calculate_sizes(dataset_name, gt_sizes, predictions_sizes)
    hypothesis_test_object_size(dataset_name, gt_sizes, predictions_sizes, os.path.join(save_path, 't_test_size.txt'), p_val)
    
    plot_comparison_histogram(dataset_name, gt_num_changes, predictions_num_changes, os.path.join(save_path, 'hist_num_changes.png'))
    plot_comparison_histogram(dataset_name, gt_spread, predictions_spread, os.path.join(save_path, 'hist_spread.png'))
    plot_comparison_histogram(dataset_name, gt_sizes, predictions_sizes, os.path.join(save_path, 'hist_sizes.png'))
    
    
def aggregate_distribution_histogram(dataset_name, ground_truths, predictions_dict, save_path):
        pass
    
    
    


        

