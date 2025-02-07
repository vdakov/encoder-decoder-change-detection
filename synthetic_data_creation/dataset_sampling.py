
import os 
import numpy as np 
import random
from random import randint
import pickle


'''The three functions below ensure that the dataset properties are saved and not re-calculated on every reboot of the script.'''

def save_properties(numbers, radiuses, sizes, filename='dataset_properties.pkl'):
    with open(filename, 'wb') as f:
        pickle.dump((numbers, radiuses, sizes), f)

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
        numbers = calculate_num_objects(images)
        radiuses = calculate_distances(images, {})[0]
        sizes = calculate_sizes(images, {})[0]

        # Save the properties to a file
        save_properties(numbers, radiuses, sizes, filename)
        return numbers, radiuses, sizes

def sample_dataset_properties(numbers, densities, sizes):
    std_num_buildings = np.std(numbers)
    d = random.choice(densities)
    std_radius = calculate_radius(num_buildings, np.std(densities))
    num_buildings = random.choice(numbers) 
    radius = calculate_radius(num_buildings, random.choice(densities))
    size = random.choice(sizes)
    std_size = np.std(sizes)
    
    return num_buildings, radius, size, std_num_buildings, std_radius, std_size
