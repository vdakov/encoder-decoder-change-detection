from matplotlib import pyplot as plt
import numpy as np
import cv2
import os  


def calculate_distances(dataset_name, ground_truth, predictions):
    ground_truth_distances = []
    predictions_distances = {}
    
    for img in ground_truth:
        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # distances = []
        for cnt in contours:
            M1 = cv2.moments(cnt)
            if M1['m00'] == 0:
                continue
            
            cx1 = M1['m10'] / M1['m00']
            cy1 = M1['m01'] / M1['m00']
        
            # distances_point = []
            area = cv2.contourArea(cnt)
            for cnt2 in contours:
                if cnt is cnt2:
                    continue
                M2 = cv2.moments(cnt2)
                if M2['m00'] == 0:
                    continue
                
                cx2 = int(M2['m10']/M2['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                d = calculate_distance_two_points(cx1, cy1, cx2, cy2, area)
                ground_truth_distances.append(d)
            

                
            # distances.append(np.mean(distances_point))
            
        # ground_truth_distances.append(np.mean(distances))
            
                
                
                
                
    for k in predictions.keys():
        
        for img in predictions[k]:
            predictions_distances[k] = {}
            contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            distances = []
            
            for cnt in contours:
                M1 = cv2.moments(cnt)
                cx1 = M1['m10'] / M1['m00']
                cy1 = M1['m01'] / M1['m00']
                distances_point = []
                area = cv2.contourArea(cnt)
                for cnt2 in contours:
                    if cnt is cnt2:
                        continue
                    M2 = cv2.moments(cnt2)
                    
                    cx2 = int(M2['m10']/M2['m00'])
                    cy2= int(M2['m01']/M2['m00'])
                    d = calculate_distance_two_points(cx1, cy1, cx2, cy2, area)
                    predictions_distances[k].append(d)

    
    return ground_truth_distances, predictions_distances

def calculate_distance_two_points(x1, y1, x2, y2, area):
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    
    return np.linalg.norm(a - b)


gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 
gt_distances, _ = calculate_distances('LEVIR', gt_images, {})
plt.hist(gt_distances, bins = 100)
plt.show()


