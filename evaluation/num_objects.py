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
    contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    return len(contours)


def calculate_num_objects(dataset_name, dataset):

    num_objects = []
    for img in dataset:
        num_objects.append(get_number_of_objects(img))
    return num_objects



