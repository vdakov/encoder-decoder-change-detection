import argparse
import os
import shutil
from sklearn.model_selection import train_test_split

#==================
# Script to split some of the datsets into a training, test and validation datasets. Set with the same random seed 
# for reproducible results. 
#==================


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type = str, default = os.path.join("..", "data", "HRSCD"))
    parser.add_argument("--base_path", type=str, default = os.path.join("..", "data", "HRSCD"))
    
    
    
    return parser.parse_args()


def copy_data(classes, images, dataset_path, base_path, set_name):

    for img in images: 
        all_exist = all(os.path.exists(os.path.join(dataset_path, c, img)) for c in classes)
        
        if all_exist:
            print('Success')
            for c in classes:
                source_path = os.path.join(dataset_path, c, img)
                destination_path = os.path.join(base_path, set_name, c)
                shutil.copy(source_path, destination_path)
        else:
            print(f"Image {img} does not exist in all classes. Skipping.")



def create_class_dirs(base_path, class_names):
    for class_name in class_names:
        os.makedirs(os.path.join(base_path, class_name), exist_ok=True)

if __name__ == "__main__":
    args = get_args()
    base_path = args.base_path
    dataset_path = args.dataset_path


    train_dir = os.path.join(base_path, 'train')
    val_dir = os.path.join(base_path, 'val')
    test_dir = os.path.join(base_path, 'test')


    classes = [name for name in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, name)) and name not in ["train", "val", "test"]]

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    create_class_dirs(train_dir, classes)
    create_class_dirs(val_dir, classes)
    create_class_dirs(test_dir, classes)

    images = list(filter(lambda x : ".png" in x, os.listdir(os.path.join(dataset_path, 'labels'))))

    train_images, temp_images = train_test_split(images, test_size=0.4, random_state=42)
    val_images, test_images = train_test_split(temp_images, test_size=0.5, random_state=42)

 
    copy_data(classes, train_images, dataset_path, base_path, 'train')
    copy_data(classes, val_images, dataset_path, base_path, 'test')
    copy_data(classes, test_images, dataset_path, base_path, 'val')



    