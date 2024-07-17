import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
from matplotlib import rcParams

# ===========================
# Functions used for various plots and figures throughout the experiment. All of them are set to 
# have the same font and color scheme. Feel free to extend it. 
# ===========================


rcParams['font.family'] = 'serif'
rcParams["font.family"] = "Times New Roman"
rcParams['axes.titlesize'] = 18  # Title font size
rcParams['axes.labelsize'] = 24  # Axis label font size
rcParams['xtick.labelsize'] = 20  # X tick label font size
rcParams['ytick.labelsize'] = 20  # Y tick label font size




def create_loss_accuracy_figures(train_metrics, val_metrics, test_metrics, model_name, save_path = None):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    

    train_loss = extract_metric(train_metrics, 'net_loss')
    val_loss = extract_metric(val_metrics, 'net_loss')
    test_loss = test_metrics['net_loss']

    train_accuracy = extract_metric(train_metrics, 'net_accuracy')
    val_accuracy = extract_metric(val_metrics, 'net_accuracy')
    test_accuracy = test_metrics['net_accuracy']

    train_precision = extract_metric(train_metrics, 'precision')
    val_precision = extract_metric(val_metrics, 'precision')
    test_precision = test_metrics['precision']

    train_recall = extract_metric(train_metrics, 'recall')
    val_recall = extract_metric(val_metrics, 'recall')
    test_recall = test_metrics['recall']


    train_f1 = [2 * p * r /  (p + r+ 1e-10) for p, r in zip(train_precision, train_recall)]
    val_f1 = [2 * p * r / (p + r+ 1e-10)  for p, r in zip(val_precision, val_recall)]
    test_f1 = (2 * test_precision * test_recall) / (test_precision + test_recall + 1e-10)



    plt.suptitle(model_name)

    axs[0].plot(train_loss, label='Train', c='blue')

    axs[0].plot(val_loss, label='Validation', c='orange')
    axs[0].axhline(y=test_loss, color='red', linestyle='--', label='Test')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Loss')

    axs[1].plot(train_accuracy, label='Train', c='blue')
    axs[1].plot(val_accuracy, label='Validation', c='orange')
    axs[1].axhline(y=test_accuracy, color='red', linestyle='--', label='Test')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')

    axs[2].plot(train_f1, label='Train F1')
    axs[2].plot(val_f1, label='Validation F1')
    axs[2].axhline(y=test_precision, color='red', linestyle='--', label='Test Precision')
    axs[2].axhline(y=test_recall, color='green', linestyle='--', label='Test Recall')
    axs[2].axhline(y=test_f1, color='blue', linestyle='--', label='Test F1')
    axs[2].set_title('Precision, Recall, and F1 Curve')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Score')

    plt.subplots_adjust(left=0.15, bottom=0.15, right=0.95, top=0.95)

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, 'loss-accuracy-precision.png'))
        


def extract_metric(metric_list, key):
    return [item[key] for item in metric_list]


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
    


def plot_comparison_histogram(dataset_name, gt_sizes, predictions_sizes, save_path):
    q25, q75 = np.percentile(predictions_sizes, [25, 75])
    iqr = q75 - q25

    bin_width = 2 * iqr / np.cbrt(len(predictions_sizes))
    freedman_diaconis_bins = int(np.ceil((max(max(gt_sizes), max(predictions_sizes)) - min(min(gt_sizes), min(predictions_sizes))) / bin_width))
    num_bins = freedman_diaconis_bins
    plt.hist(gt_sizes, bins = num_bins, color = 'black', alpha=0.5)
    plt.hist(predictions_sizes, bins = num_bins, color = 'blue', alpha=0.5)
    plt.savefig(save_path)
    plt.show()
    

    
    

        

