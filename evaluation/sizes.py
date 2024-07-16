
import os
import cv2
from matplotlib import pyplot as plt
import numpy as np





def calculate_sizes(dataset_name, ground_truth, predictions):
    ground_truth_areas= []
    predictions_areas = {}
    
    for img in ground_truth:
        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        distances = []
        for cnt in contours:

            area = cv2.contourArea(cnt)
            ground_truth_areas.append(area)
            

                
                
                
                
    for k in predictions.keys():
        
        for img in predictions[k]:
            predictions_areas[k] = {}
            contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for cnt in contours:

                area = cv2.contourArea(cnt)
                ground_truth_areas.append(area)
        

    
    return ground_truth_areas, predictions_areas



def plot_size_histogram(dataset_name, sizes):
    q25, q75 = np.percentile(sizes, [25, 75])
    iqr = q75 - q25
    print(iqr)
    print(min(sizes), max(sizes))
    bin_width = 2 * iqr / np.cbrt(len(sizes))
    freedman_diaconis_bins = int(np.ceil((max(sizes) - min(sizes)) / bin_width))
    print(freedman_diaconis_bins)
    num_bins = freedman_diaconis_bins
    plt.hist(sizes, bins = num_bins)
    plt.show()


# gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 
# gt_sizes, _ = calculate_sizes('LEVIR', gt_images, {})

# plot_size_histogram('', gt_sizes)