import json
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
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


def aggregate_category_histograms(dataset_name, plot_name, aggregate_category_metrics, save_path=None):
  
    all_values = [value for d in aggregate_category_metrics for value in list(d.values())[:-1]]
    y_limit = np.max(all_values)
    
    num_categories = len(aggregate_category_metrics[0].keys()) - 1
    num_cols = int(np.ceil(num_categories / 2))
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    
    metrics = ['TP', 'FP', 'TN', 'FN']
    
    results = []
    colors = {"Early": 'blue', "Mid-Conc." :'orange', "Mid-Diff." : 'lime', "Late" :'red'}
    
    if dataset_name == "CSCD":
        titles = {"large_change_uniform": 'LCU', "large_change_non_uniform" :'LCNU', "small_change_non_uniform" : 'SCNU', "small_change_uniform" :'SCU'}
    elif dataset_name == "HRSCD":
        titles = {"No information" : "No information" , "Artificial surfaces" : "Artificial surfaces",
                  "Agricultural areas": "Agricultural areas", "Forests" : "Forests", "Wetlands" : "Wetlands", "Water": "Water"}
    elif dataset_name == "HIUCD":
        titles = {
            "Unlabeled": "Unlabeled",
            "Water": "Water",
            "Grass": "Grass",
            "Building": "Building",
            "Green house": "Green house",
            "Road": "Road",
            "Bridge": "Bridge",
            "Others": "Others",
            "Bare land": "Bare land",
            "Woodland": "Woodland"
        }
    
    for i, c in enumerate(list(aggregate_category_metrics[0].keys())[:-1]):
        
        row, col = divmod(i, num_cols)
        ax = axes[row, col]
        x = np.arange(len(metrics))  # the label locations
 
        width = 0.24
        multiplier = 0

        for model_name, category_metrics in zip(["Early", "Mid-Conc.", "Mid-Diff.", "Late"], aggregate_category_metrics):
            
            tp = category_metrics[c][0]
            fp = category_metrics[c][1]
            tn = category_metrics[c][2]
            fn = category_metrics[c][3]
            
            precision = tp / (tp + fp) if (tp + fp) != 0 else 0
            recall = tp / (tp + fn) if (tp + fn) != 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
            
            
            offset = width * multiplier
            rects = ax.bar(x + offset, category_metrics[c], width, color=colors[model_name])
            multiplier += 1

            
            ax.set_ylim(0, int(1.1 * y_limit))
            ax.set_title(titles[c])
            ax.set_xticks(x + width, metrics)

            
            results.append({
                'model_name': model_name,
                'category': c,
                'precision': precision,
                'recall': recall,
                'f1_score': f1
            })
            
    for i in range(num_categories, num_rows * num_cols):
        fig.delaxes(axes.flat[i]) 
                

    plt.tight_layout()
    plt.subplots_adjust(left=0.05, bottom=0.05, right=0.95, top=0.95)

        
    if save_path:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'aggregate_categorical.png'))
        results_df = pd.DataFrame(results)
        results_df.to_csv(os.path.join(save_path,  'metrics_results.csv'), index=False)
        
    
    plt.show()
    
    
def compare_number_of_buildings(dataset_name, plot_name, aggregate_category_metrics, save_path=None):
    colors = {"Early": 'blue', "Mid-Conc." :'orange', "Mid-Diff." : 'lime', "Late" :'red'}
    markers = {"Early": 'o', "Mid-Conc.": 's', "Mid-Diff.": '^', "Late": 'D'}
    gt_label_added = False
    
    plt.figure(figsize=(8, 5))

    for model_name, category_metrics in zip(["Early", "Mid-Conc.", "Mid-Diff.", "Late"], aggregate_category_metrics):

        values = category_metrics['num_changes']
        ground_truth = [item[0] for item in values]
        predictions = [item[1] for item in values]
        
        plt.scatter(predictions, ground_truth, c=colors[model_name], label=model_name, alpha=0.4, edgecolors='w', s=100, marker=markers[model_name])
        
    for model_name, category_metrics in zip(["Early", "Mid-Conc.", "Mid-Diff.", "Late"], aggregate_category_metrics):
        values = category_metrics['num_changes']
        ground_truth = [item[0] for item in values]
        
        if not gt_label_added:
            plt.scatter(ground_truth, ground_truth, c='black', label='GT', edgecolors='w', s=100, marker='x')
            gt_label_added = True
        else:
            plt.scatter(ground_truth, ground_truth, c='black', edgecolors='w', s=100, marker='x')
        
        
        
    plt.xlabel('# Predicted Changes')
    plt.ylabel('# Actual Changes')
    plt.tight_layout()
    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    if save_path:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'num_buildings.png'))

    
    plt.show()