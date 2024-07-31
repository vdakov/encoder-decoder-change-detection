import os
import cv2 
from matplotlib import pyplot as plt
import numpy as np

#========================
# The number of changes is measured via contour analysis, 
# and then counting the number of contours in terms of ground truth vs. predictions. 
# The library used for it is cv2, with a topological algorithm. Currently there are still some outliers and 
# it is a point of investigation whether this is owed to model accuracy or  something else.
#========================

def get_number_of_objects(img):
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)

    # Apply adaptive thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    # Apply morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)

    # Use connectedComponentsWithStats for better analysis
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)



def calculate_num_objects(dataset_name, dataset):

    num_objects = []
    for img in dataset:
        num_objects.append(get_number_of_objects(img))
    return num_objects



# gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 
# # gt_sizes, _ = calculate_num_objects('LEVIR', gt_images)


