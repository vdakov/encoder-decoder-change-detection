import csv
import numpy as np
import pandas as pd 
import os 
import ast

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

 
    filename_category = os.path.join(save_path, 'categorical_metrics.csv')
    
    

    with open(filename_category, 'w', newline='') as csv_file:
        fieldnames = categorical_metrics.keys()
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        data = [{field: value for field, value in zip(fieldnames, row)} for row in zip(*categorical_metrics.values())]
    
        
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
    filename = os.path.join(model_path, 'tables', 'categorical_metrics.csv')


    categorical_frame = pd.read_csv(filename, sep=',', encoding='utf-8')
    
    def convert_columns_to_float(df):
        for column in df.columns:
            if column != 'num_changes':  
                df[column] = pd.to_numeric(df[column], errors='coerce')
        return df
    categorical_frame = convert_columns_to_float(categorical_frame)
    
    def parse_list_column(df, column_name):
        df[column_name] = df[column_name].apply(ast.literal_eval)
        return df
    
    categorical_frame = parse_list_column(categorical_frame, 'num_changes')
    categorical_dict = categorical_frame.to_dict(orient='list')

    return categorical_dict
