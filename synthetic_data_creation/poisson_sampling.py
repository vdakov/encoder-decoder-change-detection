import math
import numpy as np 
import random 
from random import randint 



def calculate_radius(n, density, beta = 0.05 ):
    return np.divide(np.log(np.divide(n, density)), beta) #assumes equal density between every building in image 

def poisson_disk_sampling(img, num_buildings, radius, std_radius, active_points=[]):
    '''
    Algorithm borrowed from https://sighack.com/post/poisson-disk-sampling-bridsons-algorithm. 
    It samples building a radius, equal to the density of the sampled image. The result should be approximately an image with that density. This radius should be 
    one standard deviation from the given density. The algorithm is slightly modifed, as we limit the number of buldings generated, vary the radius between the points in the 
    grid directly (instead of just being [r, 2r]). Otherwise the algorithms are identical. 
    '''
    points = [] 
    N = 2
    width, height = img.shape[:2]
    cellsize = max(math.floor(radius / np.sqrt(N)), 1)
    ncells_width = math.ceil(width / cellsize) + 1
    ncells_height = math.ceil(height / cellsize) + 1
    
    grid = [[False for i in range(ncells_width)] for j in range(ncells_height)] #initialize 2D grid
    
    def insert_point(grid, p):
        x, y = p
        
        xindex = int(math.floor(x / cellsize))
        yindex = int(math.floor(y / cellsize))
        grid[xindex][yindex] = p
        
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
                    distance = np.linalg.norm(np.array(grid[i][j]) - np.array(p))
                    if distance < radius:
                        return False


        return True

    x0, y0 = random.randrange(0, width), random.randrange(height)
    insert_point(grid, (x0, y0))
    points.append((x0, y0))
    active_points.append((x0, y0))

    while len(points) < num_buildings and len(active_points) > 0:
        random_index = random.randrange(0, len(active_points))
        p = active_points[random_index]
        x, y = p
        
        found = False
        k = 0
        
        while k < 30:
            
            theta = randint(0, 360);
            new_radius = random.uniform(radius - std_radius, radius + std_radius)
            
            pnewx = x + new_radius * np.cos(np.radians(theta));
            pnewy = y + new_radius * np.sin(np.radians(theta));
            pnew = (pnewx, pnewy)
            
            if not isValidPoint(grid, cellsize, ncells_width, ncells_height, pnew, radius):
                k += 1
                continue
            
            found = True
            insert_point(grid, pnew)
            points.append(pnew) 
            active_points.append(pnew)
            break
            
        if not found:
            del active_points[random_index]

    return points

    