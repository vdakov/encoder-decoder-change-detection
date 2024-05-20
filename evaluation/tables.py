import numpy as np
import pandas as pd 
import os 

def create_tables(train_metrics, test_metrics, model_name):
    train_frame = pd.DataFrame.from_dict(train_metrics)
    test_frame = pd.DataFrame.from_dict(test_metrics, orient='index')
    

    dir_name = model_name 
    try:
        os.makedirs(os.path.join('results', 'tables', dir_name))
    except:
        print()

        

    train_filename = os.path.join('results', 'tables', dir_name, 'train_metrics.csv')
    test_filename = os.path.join('results', 'tables', dir_name, 'test_metrics.csv')

    train_frame.to_csv(train_filename, sep=',', index=False, encoding='utf-8')
    test_frame.to_csv(test_filename, sep=',', index=False, encoding='utf-8')
