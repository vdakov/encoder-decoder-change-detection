import numpy as np 
import cv2 
from random import randint
from random import sample
import os
from poisson_sampling import poisson_disk_sampling




def select_random_image_from_list(width, height, BACKGROUND_IMAGES):
    im_path_1, im_path_2 = sample(BACKGROUND_IMAGES, 2)
    image1 = cv2.imread(os.path.join('textures', im_path_1))
    image2 = cv2.imread(os.path.join('textures', im_path_2))
    image1 = cv2.resize(image1, (width, height))
    image2 = cv2.resize(image2, (width, height))

    return image1, image2

def create_base_image(width, height, num_buildings, area, std_area, radius, std_radius, BACKGROUND_IMAGES):
    '''The base image (or T1) creation function. Based on the samples it is given, it creates a completely new building-imitating image via the Poisson Sampling algorithm.'''
    img = np.zeros((width, height, 3), np.uint8)
    background_texture, building_texture = select_random_image_from_list(width, height, BACKGROUND_IMAGES)
    points = poisson_disk_sampling(img, num_buildings, radius, std_radius)
    base_obj_width, base_obj_height = np.sqrt(area), np.sqrt(area)
    std_area = max(1, std_area) 
    
    for point in points:

        obj_width = int(np.random.normal(base_obj_width, np.sqrt(std_area)))
        obj_height = int(np.random.normal(base_obj_height, np.sqrt(std_area)))

        angle = randint(0, 360)
        x0, y0 = point

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))

    t1_mask = img.astype(np.uint8)
    t1_bg_mask =  cv2.cvtColor(t1_mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    mask_inv = cv2.bitwise_not(t1_bg_mask)  

    t1_buildings = cv2.bitwise_and(building_texture, building_texture, mask=t1_bg_mask)

    t1 = cv2.add(t1_buildings, background_texture)
    
    return t1, t1_mask, points, background_texture, building_texture

def change(t1, num_changes, area, std_area, radius, std_radius, active_points, background_texture, building_texture):
    '''The function that adds all necessary changes to T1. It differs from the T1 function with the 
    fact that Poisson Sampling is passed the old active points, and that here the first image is not modified.'''

    img = cv2.cvtColor(np.zeros(t1.shape[:2], dtype=np.uint8) , cv2.COLOR_GRAY2BGR)
    points = poisson_disk_sampling(img, num_changes, radius, std_radius, active_points)
    base_obj_width, base_obj_height = np.sqrt(area), np.sqrt(area)
    std_area = max(1, std_area) 
    for point in points:
        obj_width = int(np.random.normal(base_obj_width, np.sqrt(std_area)))
        obj_height = int(np.random.normal(base_obj_height, np.sqrt(std_area)))
        angle = randint(0, 360)
        x0, y0 = point

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))

    
    t2 = cv2.bitwise_or(t1, img)
    # t2 = t2.astype(np.uint8) * 255

    label = np.logical_xor(t1, t2).astype(np.uint8) * 255
    # label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)

    # t2_mask = label.astype(np.uint8)
    t2_bg_mask =  cv2.cvtColor(label, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    new_objects_mask = cv2.bitwise_and(t2, img)
    new_objects_mask = cv2.cvtColor(new_objects_mask, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    t1_mask = cv2.cvtColor(t1, cv2.COLOR_BGR2GRAY)
    t1_buildings = cv2.bitwise_and(building_texture, building_texture, mask=t1_mask)
    t2_buildings = cv2.bitwise_and(building_texture, building_texture, mask=new_objects_mask)

    t2 = cv2.add(background_texture, t1_buildings)
    t2 = cv2.add(t2, t2_buildings)
        
    return  t2, label