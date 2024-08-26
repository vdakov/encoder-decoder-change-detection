import cv2
import numpy as np 
import random
from random import randint
import math
import copy
import os 
import csv
import shutil

from distance import calculate_distances
from num_objects import calculate_num_objects
from sizes import calculate_sizes

def give_random_colors():
    
    color1 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    # Ensure the two colors are different
    while color2 == color1:
        color2 = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    
    return color1, color2





def create_base_image(width, height):
    img = np.zeros((width, height, 3), np.uint8)
    
    for _ in range(randint(0, 12)):

        x0, y0 = randint(0, width), randint(0, height)
        obj_width = randint(int(0.05 * width), int(0.5 * width) )
        obj_height = randint(int(0.05 * height), int(0.5 * height))
        angle = randint(0, 360)

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

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
    
    index_height = random_grid_index // rows
    index_width = random_grid_index  % rows
    
    x0 = index_width * grid_width
    y0 = index_height * grid_height
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
        x0, y0 = randint(left, right), randint(top, bot)
        obj_width = randint(int(0.1 * width_constraint), int(0.5 * width_constraint) )
        obj_height = randint(int(0.1 * height_constraint), int(0.5 * height_constraint))
        angle = randint(0, 360)

        rect = ((x0, y0), (obj_width, obj_height), angle)
        box = cv2.boxPoints(rect)
        box = np.intp(box)

        cv2.polylines(img, [box] ,True,(255,255,255))
        cv2.fillPoly(img, [box], (255, 255, 255))


        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (5, 5))
        if not small and uniform:
            img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
            img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    if small and uniform:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    else :
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)) #consider why this works
        
    if not small and uniform:
        img = cv2.morphologyEx(img, cv2.MORPH_ERODE, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel) 

    return img





def large_changes(img, uniform=True):
    num_changes = randint(0, 5)

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
            curr = spread_objects_in_range(np.zeros(t1.shape), uniform, False, math.ceil(num_changes / n_grids), x0, y0, x3, y3)
            t2 = np.logical_or(t2, curr)
        change = 'large_change_non_uniform'

    t2 = np.logical_or(t1, t2)
        
    t1 = t1.astype(np.uint8)
    t2 = t2.astype(np.uint8) * 255
    

    
    label = np.logical_xor(t1, t2).astype(np.uint8) * 255

    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    num_changes = len(cv2.findContours(label, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0])

    return t1, t2, label, change, num_changes

def small_change(img, uniform=True):
    num_changes = randint(5, 25)

    t1 = copy.deepcopy(img)
    width, height = img.shape[:2]
    change = 'small_change_uniform'

    if uniform: 
        t2 = spread_objects_in_range(np.zeros(t1.shape), uniform, True, num_changes, 0, 0, width, height)
        t2 = np.logical_or(t1, t2)
    else: 
        t2 = np.zeros(t1.shape)
        
        n_grids = randint(1, 3)
        for _ in range(n_grids):
            x0, y0, x3, y3 = give_random_grid(t2, 3, 3)
            curr = spread_objects_in_range(np.zeros(t1.shape), uniform, True, math.ceil(num_changes / n_grids), x0, y0, x3, y3)
            t2 = np.logical_or(curr, t2) 
        change = 'small_change_non_uniform'
        
    t2 = np.logical_or(t1, t2)
    t1 = t1.astype(np.uint8)
    t2 = t2.astype(np.uint8) * 255

    label = np.logical_xor(t1, t2).astype(np.uint8) * 255


    label = cv2.cvtColor(label, cv2.COLOR_BGR2GRAY)
    num_changes = len(cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])
    

    return t1, t2, label, change, num_changes

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

def read_dataset_properties(images):
    numbers = calculate_num_objects('', images)
    densities = calculate_distances('', images, {})
    sizes = calculate_sizes('', images, {})
    
    return numbers, densities, sizes 

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

def poisson_disk_sampling(num_buildings, img, area, radius, std_radius):
    '''
    Algorithm borrowed from https://sighack.com/post/poisson-disk-sampling-bridsons-algorithm. 
    It samples building a radius, equal to the density of the sampled image. The result should be approximately an image with that density. This radius should be 
    one standard deviation from the given density.
    '''
    points = [] 
    active = []
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
    
    while len(active) > 0:
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

numbers, radiuses, sizes = read_dataset_properties(gt_images)

sets = ['train', 'test', 'val']
sizes = []

sizes = [2048, 512, 512]
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
            name = f'{change}-uniform-t1.png'
            cv2.imwrite(os.path.join(t1_dir, name), time_1) 
            time_1, time_2, label, situation, num_changes = change(time_1)

            bg_color, building_color = give_random_colors()
            

            time_1 = recolor_img(time_1, bg_color, building_color)
            time_2 = recolor_img(time_2, bg_color, building_color)
            time_1 = cv2.blur(time_1,(5,5))
            time_2 = cv2.blur(time_2,(5,5))
            
  
            
            im_name = f'{set}-{j}.png'
            cv2.imwrite(os.path.join(t1_dir, im_name), time_1) 
            cv2.imwrite(os.path.join(t2_dir, im_name), time_2) 
            cv2.imwrite(os.path.join(label_dir, im_name), label) 
            csv_data.append([im_name, situation, num_changes])
            j+=1


        for _ in range(sizes[i]):

            time_1 = create_base_image(128, 128)
            time_1, time_2, label, situation, num_changes = change(time_1, uniform=False)


            bg_color, building_color = give_random_colors()


            time_1 = recolor_img(time_1, bg_color, building_color)
            time_2 = recolor_img(time_2, bg_color, building_color)
            time_1 = cv2.blur(time_1,(5,5))
            time_2 = cv2.blur(time_2,(5,5))



            im_name = f'{set}-{j}.png'
            csv_data.append([im_name, situation, num_changes])
            cv2.imwrite(os.path.join(t1_dir, im_name), time_1) 
            cv2.imwrite(os.path.join(t2_dir, im_name), time_2) 
            cv2.imwrite(os.path.join(label_dir, im_name), label) 

            j+=1

    csv_path = os.path.join(set_path, 'situation_labels.csv')
    with open(csv_path, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['im_name', 'situation', 'num_changes'])
        csv_writer.writerows(csv_data)


cscd_dir = os.path.join('..', 'data', 'CSCD')
zip_filename = os.path.join('..', 'data', 'CSCD')
shutil.make_archive(base_name=zip_filename, format='zip', root_dir=cscd_dir)




