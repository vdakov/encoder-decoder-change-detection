import cv2 
import numpy as np


def get_number_of_objects(img):
    contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return len(contours)