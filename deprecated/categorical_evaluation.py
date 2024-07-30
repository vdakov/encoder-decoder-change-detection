import json
import os
import cv2
import numpy as np
from scipy.stats import median_abs_deviation


    # DEPRECATED PARTS OF Experiment.py
    # if os.path.exists(f'{model_path}.pth') and restore_prev == True:
    #     print('Restored weights!')
    #     state_dict = torch.load(f'{model_path}.pth')
    #     net.load_state_dict(state_dict)
        
    #     training_metrics, validation_metrics, test_metrics = load_metrics(model_path)
        
    #     if not generate_plots:
    #         categorical_metrics = load_categorical_metrics(model_path)
    #         aggregate_categorical.append(categorical_metrics)
    # (...)
    #    if dataset_name == "CSCD" and generate_plots:
    #         categorical_metrics = evaluate_categories(dataset_name, test_set, predictions, ["large_change_uniform", "large_change_non_uniform", "small_change_non_uniform", "small_change_uniform"])
    #         aggregate_categorical.append(categorical_metrics)
    #         create_categorical_tables(categorical_metrics, os.path.join(model_path, 'tables'))
    #     if dataset_name == "HRSCD"and generate_plots:
    #         categorical_metrics = evaluate_categories(dataset_name, test_set, predictions, ["No information", "Artificial surfaces", "Agricultural areas", 
    #                                                                                 "Forests", "Wetlands", "Water"])
    #         aggregate_categorical.append(categorical_metrics)
    #         create_categorical_tables(categorical_metrics, os.path.join(model_path, 'tables'))
    #     if dataset_name == "HIUCD"and generate_plots:
    #         categorical_metrics = evaluate_categories(dataset_name, test_set, predictions,["Unlabeled", "Water", "Grass", "Building", "Green house", 
    #                                                                                 "Road", "Bridge", "Others", "Bare land", "Woodland"])
    #         aggregate_categorical.append(categorical_metrics)
    #         create_categorical_tables(categorical_metrics, os.path.join(model_path, 'tables'))
    #     if dataset_name == "LEVIR" and generate_plots:
    #         categorical_metrics = evaluate_categories(dataset_name, test_set, predictions, [])
    #         aggregate_categorical.append(categorical_metrics)
    #         create_categorical_tables(categorical_metrics, os.path.join(model_path, 'tables'))
    # (...)
    # if generate_plots:
    #     with open(os.path.join(experiment_path, 'aggregate_categorical.csv'), 'w', newline='') as csv_file:
    #         fieldnames = aggregate_categorical[0].keys()
    #         writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    #         writer.writeheader()
    #         writer.writerows(aggregate_categorical)
    # (...)
    # if dataset_name == "CSCD" or dataset_name == "HRSCD" or dataset_name == "HIUCD":
    #     aggregate_category_histograms(dataset_name, 'Aggregate Categorical', aggregate_categorical, os.path.join(experiment_path))
    # compare_number_of_buildings(dataset_name, '# Predicted Buildings', aggregate_categorical, os.path.join(experiment_path))

def cluster_image_colors(img, dataset_name, categories):
    ''''
    The image cagteogories are numbers betweeen 0 and 255. We can map them to colors for each category if we 
    map appropriate increasing colors to them.'''
    if dataset_name == 'HRSCD':
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
    else: 
        Z = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = len(categories)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    out = res.reshape((img.shape))

    return out

def map_to_categorical(img):
    '''
    The categorical mages are based on the indices within them. Also indexes each category.'''
    vals = np.sort(np.unique(img))

    value_to_position = {value: index for index, value in enumerate(vals)}

    positions = np.vectorize(value_to_position.get)(np.ndarray.flatten(img))

    return positions.reshape(img.shape)


def evaluate_img_categorically(y, y_hat, num_changes, y_category, categories, dataset_name, IOU_THRESHOLD=0.5):
    ''''
    Utility funcion to get all images equally evaluated in the categorical loop.
    '''


    out = {c: [0, 0, 0, 0] for c in categories}

    num_changes_predicted = len(cv2.findContours(y_hat.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])
    out['num_changes'] = [num_changes, num_changes_predicted]

    if dataset_name == "HRSCD":
        y_category = y_category[:, :, 0].astype(np.uint8)
    else: 
        y_category = y_category.astype(np.uint8)
    
    y = y.flatten()
    y_hat = y_hat.flatten()
    y_category = y_category.flatten()

    for c in categories:
        if dataset_name == "CSCD" and not ((categories.index(c) + 1) in y_category):
            continue
        
        mask = y_category == categories.index(c) + 1
    
        tp = np.sum((y == 1) & (y_hat == 1) & mask)
        fp = np.sum((y == 0) & (y_hat == 1) & mask)
        tn = np.sum((y == 0) & (y_hat == 0) & mask)
        fn = np.sum((y == 1) & (y_hat == 0) & mask)

        iou = tp / (tp + fn + fp) if (tp + fn + fp) != 0 else 0.0
        prediction_made = np.sum(y_hat) > 0
        no_object = np.sum(y) == 0
        

        if iou >= IOU_THRESHOLD:
            out[c] = [1, 0, 0, 0] #tp
        elif iou < IOU_THRESHOLD and prediction_made:      
            out[c] = [0, 1, 0, 0] #fp
        elif (not prediction_made) and no_object:
            out[c] = [0, 0, 1, 0] #tn
        elif iou == 0 and (not no_object):
            out[c] = [0, 0, 0, 1]  #fn
            
    return out



def evaluate_categories(dataset_name, dataset, predictions, categories, IOU_THRESHOLD=0.5):
    categorical_metrics = {}

    for c in categories:
        categorical_metrics[c] = [0, 0, 0, 0] #tp, fp, tn,  fn

    categorical_metrics['num_changes'] = []
    
    for img_index, predicted in zip(dataset.names, predictions):
        if dataset_name == "CSCD":
            I1, I2, cm, situation, num_changes = dataset.get_img(img_index)
            categorical = np.multiply(cm, categories.index(situation) + 1)
            categorical = np.expand_dims(categorical, axis=0)
        elif dataset_name == "HRSCD" or dataset_name == "HIUCD":
            I1, I2, cm, categorical, num_changes = dataset.get_img(img_index)
            categorical = cluster_image_colors(categorical, dataset_name, categories)
            categorical = map_to_categorical(categorical)
        else:
            I1, I2, cm, _, num_changes = dataset.get_img(img_index)
            print('Not a categorical dataset')
            
        cm = np.squeeze(cm)
        cm = np.zeros_like(cm) if np.max(cm) == np.min(cm) else (cm - np.min(cm)) / (np.max(cm) - np.min(cm))     
        cm = np.where(cm < 0.5, 0, 1)
        
        if dataset_name == "CSCD" or dataset_name == "HRSCD" or dataset_name == "HIUCD" :
            curr_metrics = evaluate_img_categorically(cm, predicted, num_changes, categorical, categories, dataset_name, IOU_THRESHOLD=IOU_THRESHOLD)

            for c in categories:
                categorical_metrics[c] = np.add(categorical_metrics[c], curr_metrics[c])
        else: 
            curr_metrics = {'num_changes':[num_changes, 
                                           len(cv2.findContours(predicted.astype(np.uint8),
                                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])]}
            
        categorical_metrics['num_changes'].append(curr_metrics['num_changes'])
        
    b_values = np.array([item[1] for item in categorical_metrics['num_changes']])

    median= np.median(b_values)
    mad = median_abs_deviation(b_values)
    

    filtered_num_changes = [item if abs(item[1] - median) <= 3 * mad else [item[0], 0] for item in categorical_metrics['num_changes']]
    categorical_metrics['num_changes'] = filtered_num_changes
    

    return categorical_metrics


def create_categorical_tables(categorical_metrics, save_path=None):
    '''
    Once we have the categorical metrics, we can use this function to store them as a JSON. It is 
    picked as an alternative format to CSV due to its better compatibility with the "number of changes" metric
    predicted. 
    
    '''

    filename_category = os.path.join(save_path, 'categorical_metrics.json')
    
    for key, value in categorical_metrics.items():
        if isinstance(value, np.ndarray):
            categorical_metrics[key] = value.tolist()

    with open(filename_category, 'w') as f:
        json.dump(categorical_metrics, f, indent=4)
