import os
import cv2 
from matplotlib import pyplot as plt
import numpy as np


def get_number_of_objects(img, min_area=50):
    '''
    The function that measures the number of changes is measured via contour analysis, 
    and then counting the number of contours in terms of ground truth vs. predictions. 
    Currently there are still some outliers and  it is a point of investigation whether this is owed to model accuracy or something else.
    The contour algorithm is from OpenCV. There is a sequence of a gaussian blur to tackle noise, thresholding, a small morpholoical operations and 
    contour finding.  
    '''
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)

    # Apply adaptive thresholding
    _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    

    # Apply morphological operations to remove small noise
    kernel = np.ones((3, 3), np.uint8)
    cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(cleaned_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_area]

    return min(len(filtered_contours), 15)



def calculate_num_objects(dataset):
    return [get_number_of_objects(img) for img in dataset]


