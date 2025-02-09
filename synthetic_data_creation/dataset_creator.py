import sys
import os 
import cv2
import numpy as np 
import random
from random import randint
from random import sample
import math
import csv
import shutil
import seaborn as sns
import matplotlib.pyplot as plt
from data_augmentation import *
from poisson_sampling import  *
from dataset_sampling import *
from change_creation import *


current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)

from evaluation.distance import calculate_distances
from evaluation.num_objects import calculate_num_objects
from evaluation.sizes import calculate_sizes


'''The CSCD Creation Procedure - where all synthetic images for CSCD are created. 


A brief overview of the procedure: 

1. Analysis of the LEVIR-CD dataset, to extract its distributions in building area, 
number of buildings, and distance between buildings (inter-connectednes). Conversion of Euclidean distance 
into radiuses around a building for a later algorithm.
2. Specification of the sizes of the desired training, validation and test datasets, as well as image size.
3. Image creation starts. 

Every image is created as such:
    - radius, number of buildings and average image area are sampled, along with the variances of the 
    distribution 
    - an empty image of the desired size is created 
    - the image is populated with the sampled number of buildings via the labels of the changes, and following a modified version of the Poisson disk sampling
    algorithm, where samples are distributed in a radius of varying size around a grid - this gives coordinates of centroids at which buildings will 
    be; the buildings' area are within a range of the fixed sampled area and one standard deviation; the images are rotated at a random angle 
        - This creates the first image T1 
    - this is repeated for the image T2 
    - the change label is the XOR between the two images 
    - The two images T1 and T2 are mapped to two random colors for their foreground and background and 
    adjusted via a small distortion to their brightness and Gaussian blur. 
    - T1, T2 and their label area created. 
    
'''


BACKGROUND_IMAGES = os.listdir('textures')


'''================================================CREATION STARTS HERE================================================'''
base_imgs = []
time_1_imgs = []
time_2_imgs = []
labels = []
situation_label = []


gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 


numbers, radiuses, sizes = read_dataset_properties(gt_images)

sets = {'train' : 2048, 'test' : 512, 'val': 512}
width = 128
height = 128
area = width * height
sizes = np.multiply(sizes, np.ones(len(sizes)) * area)
radiuses = np.multiply(radiuses, np.ones(len(radiuses)) * area)
std_radius = np.sqrt(np.var(radiuses))
std_area = np.sqrt(np.var(sizes))


# https://github.com/abin24/Textures-Dataset

dataset_name = 'CSCD-Textures'
num_imgs = [sets['train'], sets['test'], sets['val'],]
os.makedirs(os.path.join('..', 'data', dataset_name), exist_ok=True)
for set in sets.keys():
    set_path = os.path.join('..', 'data', dataset_name, set)
    t1_dir = os.path.join(set_path, 'A')
    t2_dir = os.path.join(set_path, 'B')
    label_dir = os.path.join(set_path, 'label')

    os.makedirs(set_path, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    
    j = 0

    for j in range(int(sets[set])):
    

        time_1, time_1_mask,  active_points, background_texture, building_texture = create_base_image(width, height, random.choice(numbers) // 2, random.choice(sizes), std_area, random.choice(radiuses), std_radius, BACKGROUND_IMAGES)
        num_changes = int(np.random.normal(np.mean(numbers), np.std(numbers)))
        area = abs(int(np.random.normal(np.mean(radiuses), np.std(sizes))))
        radius = int(np.random.normal(np.mean(radiuses), np.std(radiuses)))
        time_2, label  = change(time_1_mask, num_changes, area, std_area, radius, std_radius, active_points, background_texture, building_texture)

        bg_color, building_color = give_random_colors()
        
        # time_1 = recolor_img(time_1, bg_color, building_color)
        # time_2 = recolor_img(time_2, bg_color, building_color)
        time_2 = adjust_brightness(time_2, random.uniform(0.9, 1.1))
        time_1 = cv2.blur(time_1,(5,5))
        time_2 = cv2.blur(time_2,(5,5))
        
        
        #UNCOMMENT IF YOU WANT TO VISUALIZE
        
        # cv2.imshow('Time 1', time_1)
        # cv2.imshow('Time 2', time_2)
        # cv2.imshow('Label', label)
        
        # # Move windows to avoid overlap (positioning them side by side)
        # cv2.moveWindow('Time 1', 100, 100)  # Move 'Time 1' window to (100, 100)
        # cv2.moveWindow('Time 2', 300, 100)  # Move 'Time 2' window to (300, 100)
        # cv2.moveWindow('Label', 500, 100)   # Move 'Label' window to (500, 100)
        
        # # Wait for a key press and move to the next set of images
        # cv2.waitKey(0)
        
        # # Close the display windows to prepare for the next images
        # cv2.destroyAllWindows()
        

        
        im_name = f'{set}-{j}.png'
        # COMMENT OUT IF YOU DO NOT WANT TO OVERWRITE OLD IMAGES 
        cv2.imwrite(os.path.join(t1_dir, im_name), time_1) 
        cv2.imwrite(os.path.join(t2_dir, im_name), time_2) 
        cv2.imwrite(os.path.join(label_dir, im_name), label) 
        
# Zips all data together. 
cscd_dir = os.path.join('..', 'data', dataset_name)
zip_filename = os.path.join('..', 'data', dataset_name)
shutil.make_archive(base_name=zip_filename, format='zip', root_dir=cscd_dir)




