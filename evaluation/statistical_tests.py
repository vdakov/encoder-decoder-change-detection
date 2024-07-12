import numpy as np 
from scipy import stats

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


        

