import cv2
import numpy as np 
import random
from random import randint
import math
    

# no location dependence, just change


def angle_from_centroid(point):
    x, y = point
    cx, cy = centroid
    return math.atan2(y - cy, x - cx)

def create_base_image()

for _ in range(1024):

    img = np.zeros((512,512,3), np.uint8)
    theta = random.uniform(0, 1) * 2 * np.pi # [0, 2pi] radians
    height, width = img.shape[:2]
    for _ in range(10):

        x0, y0 = randint(0, width), randint(0, height)
        x1 = x0 + (randint(25, 100)) 
        x2 = x0 + (randint(-10, 10) * np.cos(theta)) 
        x3 = x1 + (randint(-10, 10)) 
        
        y1 = y0 + randint(-10, 10) 
        y2 = y0 + randint(25, 100) * np.sin(theta)
        y3 = y2 + randint(-10, 10) 

        points = [(x0, y0), (x1, y1), (x2, y2), (x3, y3)]
        centroid = (sum(x for x, y in points) / len(points), sum(y for x, y in points) / len(points))
        points = sorted(points, key=angle_from_centroid)


        pts = np.array(points, np.int32)
        cv2.polylines(img,[pts],True,(255,255,255))
        cv2.fillPoly(img, [pts], (255, 255, 255))

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE , (7, 7))
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

        img = cv2.blur(img,(5,5))

    cv2.imshow('test', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


