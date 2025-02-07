import os
import shutil
from pathlib import Path

# Configuration
SOURCE_DIR = "Splited"  # Change this to your source directory
DEST_DIR = "textures"  # Change this to your destination directory
ALLOWED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}  # Allowed image formats

def ensure_dir(directory):
    """Ensure the directory exists."""
    os.makedirs(directory, exist_ok=True)

def get_images(source_dir):
    """Recursively find all images in the source directory."""
    return [f for f in Path(source_dir).rglob("*") if f.suffix.lower() in ALLOWED_EXTENSIONS]

def rename_and_move_images(source_dir, dest_dir):
    """Recursively collapse and rename images sequentially."""
    ensure_dir(dest_dir)
    images = get_images(source_dir)
    
    for index, image_path in enumerate(images, start=1):
        new_filename = f"image_{index}{image_path.suffix.lower()}"
        new_path = Path(dest_dir) / new_filename
        shutil.move(str(image_path), str(new_path))
        print(f"Moved: {image_path} -> {new_path}")
    
    print(f"Successfully moved {len(images)} images to {dest_dir}")

if __name__ == "__main__":
    rename_and_move_images(SOURCE_DIR, DEST_DIR)
