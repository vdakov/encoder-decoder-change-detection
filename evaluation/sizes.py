
import cv2
import numpy as np




def calculate_sizes(ground_truth, predictions, scale_for_img=True):
    '''
    The function used to measure the average building size (in terms of pixels area).
    The different areas are measured in the same way as the number of changes, via a connected contours 
    topological algorithm but just using the area of each contour (in pixels). Optionally, the scale_for_img
    argument is meant to identify if the user wants the areas as a fraction of the image, thus allowing comparability between sizes of
    different image sizes.
    '''
    ground_truth_areas= []
    predictions_areas = {}
    
    kernel = np.ones((3, 3), np.uint8)
    if len(ground_truth) > 0 and scale_for_img:
        width, height = ground_truth[0].shape
        img_area = width * height
    elif len(predictions) > 0 and scale_for_img: 
        first_pred = predictions['Early'][0]
        width, height = np.array(first_pred).shape
        img_area = width * height
    else:
        img_area = 1
        
    
    for img in ground_truth:
        
        blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
        _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
        
        contours, _ = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sizes = []
        for cnt in contours:

            area = cv2.contourArea(cnt) / img_area
            sizes.append(area)
        sizes = np.array(sizes)
        ground_truth_areas.append(np.mean(sizes) if len(sizes) > 0 else 0)

    for k in predictions.keys():
        predictions_areas[k] = []
        for img in predictions[k]:
            
            blurred_image = cv2.GaussianBlur(img.astype(np.uint8), (5, 5), 0)
            _, binary_image = cv2.threshold(blurred_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            cleaned_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, kernel, iterations=2)
            
            contours, _ = cv2.findContours(cleaned_image.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            

            sizes = []
            for cnt in contours:
     
                area = cv2.contourArea(cnt) / img_area
                sizes.append(area)
            sizes = np.array(sizes) 
            predictions_areas[k].append(np.mean(sizes) if len(sizes) > 0 else 0)
        
    
    return ground_truth_areas, predictions_areas



