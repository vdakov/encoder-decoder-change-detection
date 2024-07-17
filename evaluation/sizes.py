
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np





def calculate_sizes(dataset_name, ground_truth, predictions):
    ground_truth_areas= []
    predictions_areas = {}
    
    for img in ground_truth:
        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sizes = []
        for cnt in contours:

            area = cv2.contourArea(cnt)
            sizes.append(area)
        sizes = np.array(sizes)
        ground_truth_areas.append(np.mean(sizes))
            

                
                
                
                
    for k in predictions.keys():
        predictions_areas[k] = []
        for img in predictions[k]:
            
            contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            sizes = []
            for cnt in contours:

                area = cv2.contourArea(cnt)
                sizes.append(area)
            sizes = np.array(sizes)
            predictions_areas[k].append(np.mean(sizes))
        

    
    return ground_truth_areas, predictions_areas






# gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 
# gt_sizes, _ = calculate_sizes('LEVIR', gt_images, {})

# plot_size_histogram('', gt_sizes)