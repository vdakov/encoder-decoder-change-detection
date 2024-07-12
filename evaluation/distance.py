import numpy as np
import cv2 


def calculate_distances(dataset_name, ground_truth, predictions):
    ground_truth_distances = []
    predictions_distances = {}
    
    for img in ground_truth:
        contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        distances = []
        for cnt in contours:
            M1 = cv2.moments(cnt)
            cx1 = int(M1['m10']/M1['m00'])
            cy1 = int(M1['m01']/M1['m00'])
            distances_point = []
            for cnt2 in contours:
                M2 = cv2.moments(cnt2)
                
                cx2 = int(M2['m10']/M2['m00'])
                cy2= int(M2['m01']/M2['m00'])
                d = calculate_distance_two_points(cx1, cy1, cx2, cy2)
                distances_point.append(d)
            distances.append(np.mean(distances_point))
        ground_truth_distances.append(np.mean(distances))
            
                
                
    for k in predictions.keys():
        for img in ground_truth:
            predictions_distances[k] = {}
            contours, hierarchy = cv2.findContours(img.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            distances = []
            for cnt in contours:
                M1 = cv2.moments(cnt)
                cx1 = int(M1['m10']/M1['m00'])
                cy1 = int(M1['m01']/M1['m00'])
                distances_point = []
                for cnt2 in contours:
                    M2 = cv2.moments(cnt2)
                    
                    cx2 = int(M2['m10']/M2['m00'])
                    cy2= int(M2['m01']/M2['m00'])
                    d = calculate_distance_two_points(cx1, cy1, cx2, cy2)
                    distances_point.append(d)
                distances.append(np.mean(distances_point))
        predictions_distances[k].append(np.mean(distances))
    
    return ground_truth_distances, predictions_distances

def calculate_distance_two_points(x1, y1, x2, y2):
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    
    return np.linalg.norm(a - b)