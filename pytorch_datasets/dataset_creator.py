import sys
import os 
import cv2
import numpy as np 
import random
from random import randint
import math
import os 
import csv
import shutil
import pickle

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, os.pardir))
sys.path.insert(0, parent_dir)


from evaluation.distance import calculate_distances
from evaluation.num_objects import calculate_num_objects
from evaluation.sizes import calculate_sizes



def give_random_colors():
    
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Ensure the two colors are different
    while color2 == color1:
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    return color1, color2





def create_base_image(width, height, num_buildings, area, std_area, radius, std_radius):
    img = np.zeros((width, height, 3), np.uint8)
    points = poisson_disk_sampling(img, num_buildings, radius, std_radius)
    base_obj_width, base_obj_height = np.sqrt(area), np.sqrt(area)
    
    for point in points:

        
        obj_width, obj_height = randint(base_obj_width - np.sqrt(std_area), base_obj_width + np.sqrt(std_area)), randint(base_obj_height - np.sqrt(std_area), base_obj_height + np.sqrt(std_area))
        angle = randint(0, 360)
        x0, y0 = point

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    t1 = img.astype(np.uint8)
    
    return t1, points







def change(t1, num_changes, area, std_area, radius, std_radius, active_points):

    img = np.zeros(t1.shape)
    
    points = poisson_disk_sampling(img, num_changes, radius, std_radius, active_points)
    base_obj_width, base_obj_height = np.sqrt(area), np.sqrt(area)
    
    for point in points:

        
        obj_width, obj_height = randint(base_obj_width - np.sqrt(std_area), base_obj_width + np.sqrt(std_area)), randint(base_obj_height - np.sqrt(std_area), base_obj_height + np.sqrt(std_area))
        angle = randint(0, 360)
        x0, y0 = point

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5, 5))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        
    t2 = np.logical_or(t1, t2)
    t2 = t2.astype(np.uint8) * 255

    label = np.logical_xor(t1, t2).astype(np.uint8) * 255


    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
        
    
    return  t2, label

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





#assumes equal density between every building in image 
def calculate_radius(n, density, beta = 0.05 ):
    return np.divide(np.log(np.divide(n, density)), beta)

def sample_dataset_properties(numbers, densities, sizes):
    std_num_buildings = np.std(numbers)
    d = random.choice(densities)
    std_radius = calculate_radius(num_buildings, np.std(densities))
    num_buildings = random.choice(numbers) 
    radius = calculate_radius(num_buildings, random.choice(densities))
    size = random.choice(sizes)
    std_size = np.std(sizes)
    
    return num_buildings, radius, size, std_num_buildings, std_radius, std_size

def poisson_disk_sampling(img, num_buildings, radius, std_radius, active=[]):
    '''
    Algorithm borrowed from https://sighack.com/post/poisson-disk-sampling-bridsons-algorithm. 
    It samples building a radius, equal to the density of the sampled image. The result should be approximately an image with that density. This radius should be 
    one standard deviation from the given density.
    '''
    points = [] 
    N = 2
    width, height = img.shape
    
    cellsize = math.floor(radius/np.sqrt(N));
    ncells_width = math.ceil(width/cellsize) + 1
    ncells_height = math.ceil(height/cellsize) + 1
    
    grid = [[False for i in range(ncells_width)] for j in range(ncells_height)] #initialize 2D grid
    
    def insert_point(grid, p):
        x, y = p
        xindex = int(math.floor(x / cellsize))
        yindex = int(math.floor(y / cellsize))
        grid[xindex, yindex] = p
        
    def isValidPoint(grid, cellsize, gwidth, gheight, p, radius):
                # Make sure the point is on the screen 
        x, y = p
        if (x < 0 or x >= width or y < 0 or y >= height):
            return False

        xindex = math.floor(x / cellsize);
        yindex = math.floor(y / cellsize);
        i0 = max(xindex - 1, 0);
        i1 = min(xindex + 1, gwidth - 1);
        j0 = max(yindex - 1, 0);
        j1 = min(yindex + 1, gheight - 1);

        for i in range(i0, i1 + 1):
            for j in range(j0, j1 + 1):
                if grid[i][j] is not None:
                    #euclidean distance
                    if np.linalg.norm(grid[i][j].x, grid[i][j].y, p.x, p.y) < radius:
                        return False


        return True

    
    x0, y0 = random(width), random(height)
    insert_point(grid, (x0, y0))
    points.append((x0, y0))
    active.append((x0, y0))
    
    
    while len(active) < num_buildings:
        random_index = random(len(active))
        p = active[random_index]
        x, y = p
        
        found = False
        # no K for rejection parameter as we want an exact number of points -> i.g. an "adaptation of the Poisson disk sampling"
        while not found:
            theta = random(360);
            new_radius = random(radius - std_radius, radius + std_radius);
            pnewx = x + new_radius * np.cos(np.radians(theta));
            pnewy = y + new_radius * np.sin(np.radians(theta));
            pnew = (pnewx, pnewy)
            
            if not isValidPoint(grid, cellsize, ncells_width, ncells_height, pnew, radius):
                continue
            found = True 
            
            
    
    return points

    





gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 

def save_properties(numbers, densities, sizes, filename='dataset_properties.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((numbers, densities, sizes), f)

def load_properties(filename='dataset_properties.pkl'):
    with open(filename, 'rb') as f:
        return pickle.load(f)
    
def read_dataset_properties(images, filename='dataset_properties.pkl'):
    # Check if the properties file exists
    if os.path.exists(filename):
        # Load the properties if the file exists
        return load_properties(filename)
    else:
        # Compute the properties if the file does not exist
        numbers = calculate_num_objects('', images)
        densities = calculate_distances('', images, {})
        sizes = calculate_sizes('', images, {})

        # Save the properties to a file
        save_properties(numbers, densities, sizes, filename)
        return numbers, densities, sizes

numbers, radiuses, sizes = read_dataset_properties(gt_images)
std_radius = np.sqrt(np.var(radiuses))
std_area = np.sqrt(np.var(sizes))

sets = {'train' : 2048, 'test' : 512, 'val': 512}
sizes = []

num_imgs = [2048, 512, 512]
os.makedirs(os.path.join('..', 'data', 'CSCD'), exist_ok=True)
for set in enumerate(sets.keys()):
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

    for _ in range(int(sets[set])):
    

        time_1, active_points = create_base_image(128, 128, random.choice(numbers), random.choice(sizes), std_area, random.choice(radiuses), std_radius)
        num_changes = random.choice(numbers)
        area = random.choice(sizes)
        radius = random.choice(radiuses)
        time_2, label  = change(time_1, num_changes, area, std_area, radius, std_radius, active_points)

        bg_color, building_color = give_random_colors()
        
        time_1 = recolor_img(time_1, bg_color, building_color)
        time_2 = recolor_img(time_2, bg_color, building_color)
        time_1 = cv2.blur(time_1,(5,5))
        time_2 = cv2.blur(time_2,(5,5))
        
        cv2.imshow('Time 1', time_1)
        cv2.imshow('Time 2', time_2)
        cv2.imshow('Label', label)
        
        # Move windows to avoid overlap (positioning them side by side)
        cv2.moveWindow('Time 1', 100, 100)  # Move 'Time 1' window to (100, 100)
        cv2.moveWindow('Time 2', 300, 100)  # Move 'Time 2' window to (300, 100)
        cv2.moveWindow('Label', 500, 100)   # Move 'Label' window to (500, 100)
        
        # Wait for a key press and move to the next set of images
        cv2.waitKey(0)
        
        # Close the display windows to prepare for the next images
        cv2.destroyAllWindows()
        

        
        im_name = f'{set}-{j}.png'
        # cv2.imwrite(os.path.join(t1_dir, im_name), time_1) 
        # cv2.imwrite(os.path.join(t2_dir, im_name), time_2) 
        # cv2.imwrite(os.path.join(label_dir, im_name), label) 
        csv_data.append([im_name, 'none', num_changes])
        
    




    csv_path = os.path.join(set_path, 'situation_labels.csv')
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['im_name', 'situation', 'num_changes'])
        csv_writer.writerows(csv_data)


cscd_dir = os.path.join('..', 'data', 'CSCD')
zip_filename = os.path.join('..', 'data', 'CSCD')
shutil.make_archive(base_name=zip_filename, format='zip', root_dir=cscd_dir)




