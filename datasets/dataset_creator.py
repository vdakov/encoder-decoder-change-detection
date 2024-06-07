import cv2
import numpy as np 
import random
from random import randint
import math
import copy
import os 
import csv

def give_random_color(): 
    a = 0
    b = 255
    o = 1
    R = random.randrange(a, b, o) # last value is step (optional) 
    B = random.randrange(a, b, o) 
    G = random.randrange(a, b, o) 

    return (R, G, B)
    



def create_base_image(width, height):
    img = np.zeros((width, height, 3), np.uint8)
    
    for _ in range(randint(0, 12)):

        x0, y0 = randint(0, width), randint(0, height)
        obj_width = randint(int(0.05 * width), int(0.5 * width) )
        obj_height = randint(int(0.05 * height), int(0.5 * height))
        angle = randint(0, 360)

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)


        

    return img

def give_random_grid(img, rows, columns):
    width, height = img.shape[:2]

    grid_width = int(width // columns)
    grid_height = int(height // rows)

    random_grid_index = randint(0, rows * columns - 1)
    
    x0 = (random_grid_index % columns) * grid_width
    y0 = (random_grid_index % rows) * grid_height
    x3 = x0 + grid_width - 1
    y3 = y0 + grid_height - 1
    


    return x0, y0, x3, y3

def spread_objects_in_range(img, uniform, small, num_changes, left, top, right, bot):
    width = right - left 
    height = bot - top

    if small: 
        width_constraint = width / 4
        height_constraint = height / 4
    else :
        width_constraint = width 
        height_constraint = height 

    

    for _ in range(num_changes):
        x0, y0 = randint(0, width), randint(0, height)
        obj_width = randint(int(0.1 * width_constraint), int(0.5 * width_constraint) )
        obj_height = randint(int(0.1 * height_constraint), int(0.5 * height_constraint))
        angle = randint(0, 360)

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if small and uniform:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    else :
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) #consider why this works
    img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 

    return img





def large_changes(img, uniform=True):
    num_changes = randint(1, 5)

    t1 = copy.deepcopy(img)
    
    width, height = img.shape[:2]
    change = 'large_change_uniform'

    if uniform: 
        t2 = spread_objects_in_range(np.zeros(t1.shape), uniform, False, num_changes, 0, 0, width, height)
        
    else: 
        t2 = np.zeros(t1.shape)
        n_grids = randint(2, 8)
        for _ in range(n_grids):
            x0, y0, x3, y3 = give_random_grid(t2, 3, 3)
            curr = spread_objects_in_range(np.zeros(t1.shape), uniform, False, max(num_changes // n_grids, 1), x0, y0, x3, y3)
            t2 = np.logical_or(t2, curr)
        change = 'large_change_non_uniform'

    t2 = np.logical_or(t1, t2)
        
    t1 = t1.astype(np.uint8)
    t2 = t2.astype(np.uint8) * 255
    

    
    label = np.logical_xor(t1, t2).astype(np.uint8) * 255
    label = cv2.morphologyEx(label, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return t1, t2, label, change

def small_change(img, uniform=True):
    num_changes = randint(5, 25)

    t1 = copy.deepcopy(img)
    width, height = img.shape[:2]
    change = 'small_change_uniform'

    if uniform: 
        t2 = spread_objects_in_range(np.zeros(t1.shape), uniform, True, num_changes, 0, 0, width, height)
        t2 = np.logical_or(t1, t2)
    else: 
        t2 = np.logical_or(t1, t2)
        n_grids = randint(1, 3)
        for _ in range(n_grids):
            x0, y0, x3, y3 = give_random_grid(t2, 3, 3)
            curr = spread_objects_in_range(np.zeros(t1.shape), uniform, True, max(num_changes//n_grids, 1), x0, y0, x3, y3)
            t2 = np.logical_or(curr, t2) 
        change = 'small_change_non_uniform'
        

    t1 = t1.astype(np.uint8)
    t2 = t2.astype(np.uint8) * 255

    label = np.logical_xor(t1, t2).astype(np.uint8) * 255
    label = cv2.morphologyEx(label, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))

    return t1, t2, label, change

def recolor_img(img, bg_color, building_color):
    black_pixels = np.where(
        (img[:, :, 0] == 0) & 
        (img[:, :, 1] == 0) & 
        (img[:, :, 2] == 0)
    )
    img[black_pixels] = bg_color

    white_pixels = np.where(
        (img[:, :, 0] == 255) & 
        (img[:, :, 1] == 255) & 
        (img[:, :, 2] == 255)
    )
    img[white_pixels] = building_color
    
    return img

base_imgs = []
time_1_imgs = []
time_2_imgs = []
labels = []
situation_label = []

change_methods = [large_changes, small_change]
sets = ['train', 'test', 'val']
# sizes = [608, 208, 208]
sizes = [64, 32, 32]

os.makedirs(os.path.join('..', 'data', 'CSCD'), exist_ok=True)
for i, set in enumerate(sets):
    set_path = os.path.join('..', 'data', 'CSCD', set)
    t1_dir = os.path.join(set_path, 'A')
    t2_dir = os.path.join(set_path, 'B')
    label_dir = os.path.join(set_path, 'label')

    os.makedirs(set_path, exist_ok=True)
    os.makedirs(t1_dir, exist_ok=True)
    os.makedirs(t2_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    csv_data = []
    
    j = 0
    for change in change_methods: 
        for _ in range(sizes[i]):
            time_1 = create_base_image(128, 128)
            time_1, time_2, label, situation = change(time_1)

            bg_color = give_random_color()
            building_color = give_random_color()
            

            time_1 = recolor_img(time_1, bg_color, building_color)
            time_2 = recolor_img(time_2, bg_color, building_color)
            time_1 = cv2.blur(time_1,(5,5))
            time_2 = cv2.blur(time_2,(5,5))
            
  
            
            im_name = f'{set}-{j}.png'
            cv2.imwrite(os.path.join(t1_dir, im_name), time_1) 
            cv2.imwrite(os.path.join(t2_dir, im_name), time_2) 
            cv2.imwrite(os.path.join(label_dir, im_name), label) 
            csv_data.append([im_name, situation])
            j+=1


        for _ in range(sizes[i]):

            time_1 = create_base_image(128, 128)
            time_1, time_2, label, situation = change(time_1, uniform=False)


            bg_color = give_random_color()
            building_color = give_random_color()


            time_1 = recolor_img(time_1, bg_color, building_color)
            time_2 = recolor_img(time_2, bg_color, building_color)
            time_1 = cv2.blur(time_1,(5,5))
            time_2 = cv2.blur(time_2,(5,5))



            im_name = f'{set}-{j}.png'
            csv_data.append([im_name, situation])
            cv2.imwrite(os.path.join(t1_dir, im_name), time_1) 
            cv2.imwrite(os.path.join(t2_dir, im_name), time_2) 
            cv2.imwrite(os.path.join(label_dir, im_name), label) 

            j+=1

    csv_path = os.path.join(set_path, 'situation_labels.csv')
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['im_name', 'situation'])
        csv_writer.writerows(csv_data)


  





