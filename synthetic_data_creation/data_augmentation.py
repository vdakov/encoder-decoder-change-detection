import random
import numpy as np
import cv2


def give_random_colors():
    '''The function that returns two different foreground and background colors out of the entire RGB pallete.'''
    
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Ensure the two colors are different
    while color2 == color1:
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    return color1, color2

def adjust_brightness(image, factor):
    '''Does the color mutation between T1 and T2 - the most suitable way found was via an HSV brightness adjustment. Mean to add more invariance to the dataset.'''
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

     
    return image

def recolor_img(img, bg_color, building_color):
    '''The function that maps the black and white image's foreground and background to specifically chosen colors.'''
    # black_pixels = np.where(
    #     (img[:, :, 0] == 0) & 
    #     (img[:, :, 1] == 0) & 
    #     (img[:, :, 2] == 0)
    # )
    # img[black_pixels] = bg_color

    white_pixels = np.where(
        (img[:, :, 0] == 255) & 
        (img[:, :, 1] == 255) & 
        (img[:, :, 2] == 255)
    )
    img[white_pixels] = building_color
    
    return img
