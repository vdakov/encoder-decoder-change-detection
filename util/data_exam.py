import cv2 
import os 
import numpy as np 

dirname = os.path.join('..', 'data', 'HiUCD-Mini')
set_name = 'train'
name = '583536_8.png'

label = cv2.imread(os.path.join(dirname,set_name, "label", name))
A = cv2.imread(os.path.join(dirname,set_name,"A", name))
B = cv2.imread(os.path.join(dirname,set_name,"B", name))

max_height = 400 
max_width = 400


print(np.unique(label[:, :, 0]))
print(np.unique(label[:, :, 1]))
print(np.unique(label[:, :, 2]))



print(label.shape)


def resize_image(image, max_height, max_width):
    height, width = label.shape[:2] 

    if height > max_height or width > max_width:
        scaling_factor = min(max_width / width, max_height / height)
        return cv2.resize(image, (int(width * scaling_factor), int(height * scaling_factor)))
    return image

A = resize_image(A, max_height, max_width)
B = resize_image(B, max_height, max_width)
gt = resize_image(label, max_height, max_width)[:, :, 0]
gt = np.where(gt < 2, 0, 255).astype(np.uint8)




cv2.imshow('A', A) 
cv2.imshow('B', B) 
cv2.imshow('0', gt) 
cv2.waitKey()   