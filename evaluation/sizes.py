
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np


#====================
# Size is measured in the same way as the number of changes, via a connected contours 
# topological algorithm but just using the area of each contour (in pixels). 
#====================



def calculate_sizes(dataset_name, ground_truth, predictions):
    ground_truth_areas= []
    predictions_areas = {}
    
    kernel = np.ones((3, 3), np.uint8)
    
    for img in ground_truth:
        blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, hierarchy = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sizes = []
        for cnt in contours:

            area = cv2.contourArea(cnt)
            sizes.append(area)
        sizes = np.array(sizes)
        ground_truth_areas.append(np.mean(sizes) if len(sizes) > 0 else 0)
            

                
                
                
                
    for k in predictions.keys():
        predictions_areas[k] = []
        for img in predictions[k]:
            
            blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
            _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, hierarchy = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            

            sizes = []
            for cnt in contours:

                area = cv2.contourArea(cnt)
                sizes.append(area)
            sizes = np.array(sizes)
            predictions_areas[k].append(np.mean(sizes) if len(sizes) > 0 else 0)
        

    
    return ground_truth_areas, predictions_areas






# gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 
# gt_sizes, _ = calculate_sizes('LEVIR', gt_images, {})

# plot_size_histogram('', gt_sizes)