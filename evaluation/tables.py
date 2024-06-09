import numpy as np
import pandas as pd 
import os 

def create_tables(train_metrics, val_metrics, test_metrics, model_name, save_path=None):
    train_frame = pd.DataFrame.from_dict(train_metrics)
    val_frame = pd.DataFrame.from_dict(val_metrics)
    test_frame = pd.DataFrame.from_dict(test_metrics, orient='index')

    if save_path:
        os.makedirs(save_path, exist_ok=True)

        train_filename = os.path.join(save_path, 'train_metrics.csv')
        val_filename = os.path.join(save_path, 'train_metrics.csv')
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


def load_metrics(model_path):
    train_filename = os.path.join(model_path, 'tables', 'train_metrics.csv')
    val_filename = os.path.join(model_path, 'tables', 'val_metrics.csv')
    test_filename = os.path.join(model_path, 'tables', 'test_metrics.csv')

    train_frame = pd.read_csv(train_filename, sep=',', encoding='utf-8')
    val_frame = pd.read_csv(val_filename, sep=',', encoding='utf-8')
    test_frame = pd.read_csv(test_filename, sep=',', encoding='utf-8')

    return train_frame, val_frame, test_frame
