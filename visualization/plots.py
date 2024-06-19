import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

from matplotlib import rcParams
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['DejaVu Serif']
rcParams['font.size'] = 24  # You can change this to the desired font size
rcParams['font.weight'] = 'bold'
rcParams['axes.titlesize'] = 24  # Title font size
rcParams['axes.titleweight'] = 'bold'  # Title font weight
rcParams['axes.labelsize'] = 24  # Axis label font size
rcParams['axes.labelweight'] = 'bold'  # Axis label font weight
rcParams['xtick.labelsize'] = 24  # X tick label font size
rcParams['ytick.labelsize'] = 12  # Y tick label font size



def create_figures(train_metrics, val_metrics, test_metrics, model_name, save_path = None):
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
    axs[0].legend()

    # Plot accuracy curves
    axs[1].plot(train_accuracy, label='Train', c='blue')
    axs[1].plot(val_accuracy, label='Validation', c='orange')
    axs[1].axhline(y=test_accuracy, color='red', linestyle='--', label='Test')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Accuracy')
    axs[1].legend()




    axs[2].plot(train_f1, label='Train F1')
    axs[2].plot(val_f1, label='Validation F1')
    axs[2].axhline(y=test_precision, color='red', linestyle='--', label='Test Precision')
    axs[2].axhline(y=test_recall, color='green', linestyle='--', label='Test Recall')
    axs[2].axhline(y=test_f1, color='blue', linestyle='--', label='Test F1')
    axs[2].set_title('Precision, Recall, and F1 Curve')
    axs[2].set_xlabel('Epochs')
    axs[2].set_ylabel('Score')
    axs[2].legend()

    if save_path:
        os.makedirs(save_path, exist_ok=True)
        os.makedirs(os.path.join(save_path, model_name), exist_ok=True)


        plt.savefig(os.path.join(save_path, model_name, 'loss-accuracy-precision.png'))

    # plt.show()


def extract_metric(metric_list, key):
    return [item[key] for item in metric_list]



def category_histograms(model_name, plot_name, category_metrics, save_path=None):
    width_ratios = np.ones(len(category_metrics.keys()))
    width_ratios[-1] = 3

    fig, ax = plt.subplots(1, len(category_metrics.keys()), figsize=(21, 5), width_ratios=width_ratios)

    metrics = ['tp', 'fp', 'tn', 'fn']
    

    for i, c in enumerate(list(category_metrics.keys())[:-1]):
        
        tp = category_metrics[c][0]
        fp = category_metrics[c][1]
        tn = category_metrics[c][2]
        fn = category_metrics[c][3]
        precision = tp  / (tp + fp + 1e-10)
        recall = tp / (tp + fn+ 1e-10)
        f1 = (2 * precision * recall) / (precision + recall+ 1e-10)

        legend_labels = [
            f'Precision: {precision:.2f}',
            f'Recall: {recall:.2f}',
            f'F1 Score: {f1:.2f}'
        ]
        ax[i].set_ylim(0, 2000)
        # Create custom legend entries
        custom_lines = [plt.Line2D([0], [0], color='red', lw=2),
                        plt.Line2D([0], [0], color='orange', lw=2),
                        plt.Line2D([0], [0], color='purple', lw=2)]
        
        # ax[i].legend(custom_lines, legend_labels, loc='upper right')
        ax[i].bar(metrics, category_metrics[c])
        ax[i].set_title(c)


    fig.suptitle(plot_name + "-" + model_name)
    
    values = category_metrics['num_changes']
    ground_truth = [item[0] for item in values]
    predictions = [item[1] for item in values]
    avg_diff = np.mean([np.abs(item[0] - item[1]) for item in values])
    
    
    markerline_gt, stemlines_gt, baseline_gt = ax[-1].stem(np.arange(len(ground_truth)), ground_truth, linefmt='orange', markerfmt='o', basefmt=' ', label = 'Ground Truth', use_line_collection=True)
    plt.setp(markerline_gt, 'markersize', 5)
    plt.setp(stemlines_gt, 'alpha', 0.7)
    ax[-1].stem(np.arange(len(predictions)), predictions, linefmt='blue', markerfmt='bo', basefmt=' ', label='Prediction', use_line_collection=True)
    ax[-1].axhline(y=avg_diff, color='red', linestyle='solid', label=f'Mean Diff: {avg_diff:.2f}')
    ax[-1].axhline(y=np.mean(ground_truth), color='purple', linestyle='--', label=f'Mean GT: {np.mean(ground_truth):.2f}')
    ax[-1].axhline(y=np.mean(predictions), color='cyan', linestyle='--', label=f'Mean Pred: {np.mean(predictions):.2f}')
    ax[-1].legend()
    ax[-1].set_title("Number of Changes Per Image")
    
    
    if save_path:
        plt.savefig(os.path.join(save_path, 'categorical.png'))
    plt.show()

def aggregate_category_histograms(dataset_name, plot_name, aggregate_category_metrics, save_path=None):
  
    all_values = [value for d in aggregate_category_metrics for value in list(d.values())[:-1]]
    y_limit = np.max(all_values)
    
    num_categories = len(aggregate_category_metrics[0].keys()) - 1
    num_cols = int(np.ceil(num_categories / 2))
    num_rows = 2

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 5 * num_rows))
    # fig.suptitle(plot_name, weight = 'bold')
    metrics = ['TP', 'FP', 'TN', 'FN']
    
    results = []
    colors = {"Early": 'blue', "Mid-Conc." :'orange', "Mid-Diff." : 'lime', "Late" :'red'}
    
    if dataset_name == "CSCD":
        titles = {"large_change_uniform": 'LCU', "large_change_non_uniform" :'LCNU', "small_change_non_uniform" : 'SCNU', "small_change_uniform" :'SCU'}
    elif dataset_name == "HRSCD":
        titles = {"No information" : "No information" , "Artificial surfaces" : "Artificial surfaces",
                  "Agricultural areas": "Agricultural areas", "Forests" : "Forests", "Wetlands" : "Wetlands", "Water": "Water"}
    
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
            
    #hide unused plots     
    for i in range(num_categories, num_rows * num_cols):
        fig.delaxes(axes.flat[i]) 
            
           
           
                
    custom_lines = [
        plt.Line2D([0], [0], color=colors["Early"], lw=2, label="_Early"),
        plt.Line2D([0], [0], color=colors["Mid-Conc."], lw=2, label="_Mid-Conc."),
        plt.Line2D([0], [0], color=colors["Mid-Diff."], lw=2, label="_Mid-Diff."),
        plt.Line2D([0], [0], color=colors["Late"], lw=2, label="_Late")
    ]

    # fig.legend(custom_lines, list(colors.keys()), loc='upper left')
    plt.tight_layout()
    # plt.subplots_adjust(left=0.05, right=0.05, top=0.05, bottom=0.05)

        
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
        if not gt_label_added:
            plt.scatter(ground_truth, ground_truth, c='black', label='GT', alpha=0.6, edgecolors='w', s=100, marker='x')
            gt_label_added = True
        else:
            plt.scatter(ground_truth, ground_truth, c='black', alpha=0.6, edgecolors='w', s=100, marker='x')
        
        plt.scatter(predictions, ground_truth, c=colors[model_name], label=model_name, alpha=0.6, edgecolors='w', s=100, marker=markers[model_name])
        
    plt.xlabel('# Predicted Changes')
    plt.ylabel('# Actual Changes')
    plt.title(plot_name, weight='bold')
    # plt.legend()

    if save_path:
        os.makedirs(os.path.join(save_path), exist_ok=True)
        plt.savefig(os.path.join(save_path, 'num_buildings.png'))

    
    plt.show()
    
    

        

