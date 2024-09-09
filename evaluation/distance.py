from matplotlib import pyplot as plt
import numpy as np
import cv2
import os  
from sklearn.cluster import KMeans



def calculate_distances(dataset_name, ground_truth, predictions, scale_for_img=True):
    ground_truth_distances = []
    predictions_distances = {}
    kernel = np.ones((3, 3), np.uint8)
    
    if len(ground_truth) > 0 and scale_for_img:
        width, height = ground_truth[0].shape
        img_area = width * height
    elif len(predictions.keys()) > 0 and scale_for_img: 
        first_pred = predictions['Early'][0]
        width, height = np.array(first_pred).shape
        img_area = width * height
    else:
        img_area = 1
        
    
    for img in ground_truth:
        blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            
        contours, hierarchy = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        distances = []
        for cnt in contours:
            M1 = cv2.moments(cnt)
            if M1['m00'] == 0:
                continue
            
            cx1 = M1['m10'] / M1['m00']
            cy1 = M1['m01'] / M1['m00']
        
            distances_point = []
            area = cv2.contourArea(cnt)
            for cnt2 in contours:
                if cnt is cnt2:
                    continue
                M2 = cv2.moments(cnt2)
                if M2['m00'] == 0:
                    continue
                
                cx2 = int(M2['m10']/M2['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                d = calculate_distance_two_points(cx1, cy1, cx2, cy2, area) / img_area
                distances_point.append(d)
            
            distances_point = np.array(distances_point)
            distances.append(np.mean(distances_point) if len(distances_point) > 0 else 0)
            
        if len(distances) > 0:
            
            distances = np.array(distances)
            ground_truth_distances.append(np.mean(distances))
        else: 
            ground_truth_distances.append(0)
            

                
                
                
                
    for k in predictions.keys():
        predictions_distances[k] = []
        for img in predictions[k]:
            
            blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
            _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, hierarchy = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            distances = []
            
            for cnt in contours:
                M1 = cv2.moments(cnt)
                if M1['m00'] == 0:
                    continue
                cx1 = M1['m10'] / M1['m00']
                cy1 = M1['m01'] / M1['m00']
                distances_point = []
                
                for cnt2 in contours:
                    if cnt is cnt2:
                        continue
                    M2 = cv2.moments(cnt2)
                    area = cv2.contourArea(cnt2)
                    if M2['m00'] == 0:
                        continue
                    cx2 = int(M2['m10']/M2['m00'])
                    cy2 = int(M2['m01']/M2['m00'])
                    d = calculate_distance_two_points(cx1, cy1, cx2, cy2, area) / img_area
                    distances_point.append(d)
                    
                distances_point = np.array(distances_point)
                distances.append(np.mean(distances_point) if len(distances_point) > 0 else 0)
            
                
            if len(distances) > 0:    
                distances = np.array(distances)
                predictions_distances[k].append(np.mean(distances))
            else:
                predictions_distances[k].append(0)
        

    
    return ground_truth_distances, predictions_distances

def calculate_distance_two_points(x1, y1, x2, y2, area):
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    
    return np.linalg.norm(a - b)






def calculate_connectedness(dataset_name, ground_truth, predictions, scale_for_img=True):
    '''
    =========================
    Using the centroids of each contour, we apply an urban network analysis formula. 
    This involves summing the distances between all points in the image and then taking the mean. 
    The distance per image A is measured as follows:

    D_A = (1/n) * sum_{i=1}^n sum_j (S_contour_i / e^(beta * d_ij))

    where:
    S_contour_i = area of contour i
    d_ij = Euclidean distance between point i and point j
    n = number of contours
    beta = coefficient
    ==============================
    The clustering used here is to get draw a decision boundary between close and far away changes.
    ==============================
    '''
    ground_truth_connectedness = []
    predictions_connectedness = {}
    kernel = np.ones((3, 3), np.uint8)
    
    for img in ground_truth:
        blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            
        contours, hierarchy = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        distances = []
        for cnt in contours:
            M1 = cv2.moments(cnt)
            if M1['m00'] == 0:
                continue
            
            cx1 = M1['m10'] / M1['m00']
            cy1 = M1['m01'] / M1['m00']
        
            distances_point = []
            area = cv2.contourArea(cnt)
            for cnt2 in contours:
                if cnt is cnt2:
                    continue
                M2 = cv2.moments(cnt2)
                if M2['m00'] == 0:
                    continue
                
                cx2 = int(M2['m10']/M2['m00'])
                cy2 = int(M2['m01']/M2['m00'])
                d = calculate_connectedness_two_points(cx1, cy1, cx2, cy2, area)
                distances_point.append(d)
            
            distances_point = np.array(distances_point)
            distances.append(np.sum(distances_point) if len(distances_point) > 0 else 0)
            
        if len(distances) > 0:
            
            distances = np.array(distances)
            ground_truth_connectedness.append(np.mean(distances))
        else: 
            ground_truth_connectedness.append(0)
            

                
                
                
                
    for k in predictions.keys():
        predictions_connectedness[k] = []
        for img in predictions[k]:
            
            blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
            _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, hierarchy = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            distances = []
            
            for cnt in contours:
                M1 = cv2.moments(cnt)
                if M1['m00'] == 0:
                    continue
                cx1 = M1['m10'] / M1['m00']
                cy1 = M1['m01'] / M1['m00']
                distances_point = []
                
                for cnt2 in contours:
                    if cnt is cnt2:
                        continue
                    M2 = cv2.moments(cnt2)
                    area = cv2.contourArea(cnt2)
                    if M2['m00'] == 0:
                        continue
                    cx2 = int(M2['m10']/M2['m00'])
                    cy2 = int(M2['m01']/M2['m00'])
                    d = calculate_connectedness_two_points(cx1, cy1, cx2, cy2, area)
                    distances_point.append(d)
                    
                distances_point = np.array(distances_point)
                distances.append(np.sum(distances_point) if len(distances_point) > 0 else 0)
            
                
            if len(distances) > 0:    
                distances = np.array(distances)
                predictions_connectedness[k].append(np.mean(distances))
            else:
                predictions_connectedness[k].append(0)
        
    print('Connectedness', predictions_connectedness)
    
    return ground_truth_connectedness, predictions_connectedness
    
def calculate_connectedness_two_points(x1, y1, x2, y2, area):
    a = np.array((x1, y1))
    b = np.array((x2, y2))
    
    beta = 0.05 
    
    return 1 / np.exp(beta * np.linalg.norm(a - b)) 


def plot_spread_histogram(dataset_name, distances):
    q25, q75 = np.percentile(distances, [25, 75])
    iqr = q75 - q25
    bin_width = 2 * iqr / np.cbrt(len(distances))
    freedman_diaconis_bins = int(np.ceil((max(distances) - min(distances)) / bin_width))
    num_bins = freedman_diaconis_bins
    
    
    plt.hist(distances, bins = num_bins)
    plt.show()
    
def cluster_and_plot_clusters(distances, images):
    label = KMeans(n_clusters=2, random_state=0, n_init="auto").fit_predict(np.array(distances).reshape(-1,1))
    label = np.array(label)
    gt_distances = np.array(gt_distances)
    a = gt_distances[np.where(label == 0)]
    b = gt_distances[np.where(label == 1)]

    plt.scatter(a, np.zeros_like(a), color = 'blue')
    plt.scatter(b, np.zeros_like(b), color = 'orange') 
    plt.show()

    a_indices = np.where(label == 0)[0]
    b_indices = np.where(label == 1)[0]


    display_images([images[i] for i in a_indices], "Cluster 0 Images")
    display_images([images[i] for i in b_indices], "Cluster 1 Images")







def display_images(images, title):
    plt.figure(figsize=(20, 10))
    cols = 5
    rows = min(len(images) // cols + 1, 5)
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
        print(rows, cols, i)
        plt.subplot(rows, cols, i + 1)
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.suptitle(title)
    plt.show()



# gt_images = [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'train', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'test', 'label'))] 
# gt_images = gt_images + [cv2.imread(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label', img), cv2.IMREAD_GRAYSCALE) for img in os.listdir(os.path.join('..', 'data', 'LEVIR-CD', 'val', 'label'))] 
# gt_distances, _ = calculate_distances('LEVIR', gt_images, {})
