from PIL import Image
import os
import argparse
import numpy as np 
from rasterio.plot import show
import rasterio


#================================
# For the purposes of our study, some images were far too large (e.g. the ones in LEVIR). These functions form a script
# meant to downsample the images into smaller ones, while preserving as much detail as possible and clearing them up for 
# interpolation artifacts. 
#================================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dirname", type = str, default = os.path.join("..", "data", "HRSCD", "A"))
    parser.add_argument("--resize_ratio", type=float, default = 1)
    
    
    return parser.parse_args()


def resize_aspect_fit(path, resize_ratio):
    dir = os.listdir(path)
    for item in dir:
        item_path = os.path.join(path, item)
        if '.png' in item_path:
            os.remove(item_path)

        if os.path.isfile(item_path) and item.endswith('.tif'):

    
            
        
            with rasterio.open(item_path) as src:
                img = src.read()

            
            new_image_width = int(img.shape[1] * resize_ratio)
            new_image_height = int(img.shape[2] * resize_ratio)
            image = None 

            if img.shape[0] == 1:
                # Normalize the single-band image
                band = img[0]
                band_min = band.min()
                band_max = band.max()
                if band_max != band_min:
                    image = (band - band_min) / (band_max - band_min)
                else:
                    image = band
                image = Image.fromarray((image * 255).astype(np.uint8))
                image = image.resize((new_image_width, new_image_height), Image.Resampling.LANCZOS)
            else:
                # Handle multi-band images (assuming 3-band RGB)
                image = np.dstack([img[i] for i in range(min(3, img.shape[0]))])
                image = Image.fromarray(image.astype(np.uint8))
                image = image.resize((new_image_width, new_image_height), Image.Resampling.LANCZOS)
            new_item_path = os.path.splitext(item_path)[0] + "_small.png"
            
            image.save(new_item_path, 'PNG', quality=90)
            # show(image)

if __name__ == "__main__":
    args = get_args()
    
    resize_ratio = 0.25

    # resize_aspect_fit(os.path.join( "..", "data", "HRSCD", "A"), resize_ratio)
    # resize_aspect_fit(os.path.join( "..", "data", "HRSCD", "B"), resize_ratio)
    # resize_aspect_fit(os.path.join( "..", "data", "HRSCD", "labels"), resize_ratio)
    # resize_aspect_fit(os.path.join( "..", "data", "HRSCD", "labels_land_cover_A"), resize_ratio)
    # resize_aspect_fit(os.path.join(     "..", "data", "HRSCD", "labels_land_cover_B"), resize_ratio)




