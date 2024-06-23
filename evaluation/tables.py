import csv
import numpy as np
import pandas as pd 
import os 
import ast
import json

def create_tables(train_metrics, val_metrics, test_metrics, model_name, save_path=None):
    train_frame = pd.DataFrame.from_dict(train_metrics)
    val_frame = pd.DataFrame.from_dict(val_metrics)
    test_frame = pd.DataFrame.from_dict(test_metrics, orient='index')

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        train_filename = os.path.join(save_path, 'train_metrics.csv')
        val_filename = os.path.join(save_path, 'val_metrics.csv')
        test_filename = os.path.join(save_path, 'test_metrics.csv')

        train_frame.to_csv(train_filename, sep=',', index=False, encoding='utf-8')
        val_frame.to_csv(val_filename, sep=',', index=False, encoding='utf-8')
        test_frame.to_csv(test_filename, sep=',', index=False, encoding='utf-8')

        train_filename_tex = os.path.join(save_path, 'train_metrics.tex')
        val_filename_tex = os.path.join(save_path, 'val_metrics.tex')
        test_filename_tex = os.path.join(save_path, 'test_metrics.tex')

        # Save as LaTeX
        with open(train_filename_tex, 'w', encoding='utf-8') as f:
            f.write(train_frame.to_latex(index=False))

        with open(val_filename_tex, 'w', encoding='utf-8') as f:
            f.write(val_frame.to_latex(index=False))

        with open(test_filename_tex, 'w', encoding='utf-8') as f:
            f.write(test_frame.to_latex(index=False))
            
def create_categorical_tables(categorical_metrics, model_name, save_path=None):

    filename_category = os.path.join(save_path, 'categorical_metrics.json')
    
    for key, value in categorical_metrics.items():
        if isinstance(value, np.ndarray):
            categorical_metrics[key] = value.tolist()

    with open(filename_category, 'w') as f:
        json.dump(categorical_metrics, f, indent=4)
        
    

def store_mean_difference(aggregate_category_metrics, experiment_name):
    
    data = []
    
    for model_name, categorical_metrics in zip(["Early", "Mid-Conc.", "Mid-Diff.", "Late"], aggregate_category_metrics):
        gt = [item[0] for item in categorical_metrics['num_changes']]
        pred = [item[1] for item in categorical_metrics['num_changes']]
        data.append([model_name, np.mean(np.abs(np.subtract(gt, pred)))])
    
    
    with open(os.path.join(experiment_name, 'mean_differences.csv'), 'w', newline='') as csv_file:
        
        writer = csv.writer(csv_file)
        writer.writerow(['Model Name', 'Mean Absolute Difference'])  # Write the header
        writer.writerows(data)
    
        


def load_metrics(model_path):
    train_filename = os.path.join(model_path, 'tables', 'train_metrics.csv')
    val_filename = os.path.join(model_path, 'tables', 'val_metrics.csv')
    test_filename = os.path.join(model_path, 'tables', 'test_metrics.csv')

    train_frame = pd.read_csv(train_filename, sep=',', encoding='utf-8')
    val_frame = pd.read_csv(val_filename, sep=',', encoding='utf-8')
    test_frame = pd.read_csv(test_filename, sep=',', encoding='utf-8')
    
    def convert_columns_to_float(df):
        for column in df.columns:
            df[column] = pd.to_numeric(df[column], errors='coerce')
        return df
        
    train_frame = convert_columns_to_float(train_frame)
    val_frame = convert_columns_to_float(val_frame)
    test_frame = convert_columns_to_float(test_frame)
    
    train_frame = train_frame.to_dict(orient='list')
    val_frame = val_frame.to_dict(orient='list')
    test_frame = test_frame.to_dict(orient='list')
    
    
    train_list = []
    val_list = []

    
    for i in range(len(train_frame['net_loss'])):
        train_list.append({
            'net_loss': float(train_frame['net_loss'][i]), 
            'net_accuracy': float(train_frame['net_accuracy'][i]), 
            'precision': float(train_frame['precision'][i]), 
            'recall': float(train_frame['recall'][i]), 
            'f1': float(train_frame['f1'][i])
        })


    for i in range(len(val_frame['net_loss'])):
        val_list.append({
            'net_loss': float(val_frame['net_loss'][i]), 
            'net_accuracy': float(val_frame['net_accuracy'][i]), 
            'precision': float(val_frame['precision'][i]), 
            'recall': float(val_frame['recall'][i]), 
            'f1': float(val_frame['f1'][i])
        })


    test_list = {
        'net_loss': float(test_frame['0'][0]), 
        'net_accuracy': float(test_frame['0'][1]), 
        'precision': float(test_frame['0'][2]), 
        'recall': float(test_frame['0'][3]), 
        'f1': float(test_frame['0'][4])
    }
            
            
    return train_list, val_list, test_list

def load_categorical_metrics(model_path):
    with open(os.path.join(model_path,  'tables', 'categorical_metrics.json'), 'r') as f:
        loaded_data_dict = json.load(f)
        
    return loaded_data_dict