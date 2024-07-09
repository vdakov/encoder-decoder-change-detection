import torch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
import numpy as np
import cv2
from scipy.stats import median_abs_deviation


def evaluate_net_predictions(net, criterion, dataset, IOU_THRESHOLD=0.5):
    '''A non-categorical evaluation that evaluates the accuracy of a neural networ as is. '''
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tot_loss = 0

    for img_index in dataset.names:
        I1, I2, cm, _, _ = dataset.get_img(img_index)

        I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
        I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))
        cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm), 0).long().to(device))

        output = net(I1, I2).float().to(device)

        loss = criterion(output, cm)
        tot_loss += loss.item()

        predicted =  np.exp(np.squeeze(output.cpu().detach().numpy())[1])
        predicted = np.where(predicted < 0.5, 0, 1)

        cm = np.squeeze(cm.cpu().detach().numpy())
        cm = (cm - np.min(cm)) / (np.ptp(cm)) if np.ptp(cm) != 0 else np.zeros_like(cm)
        gt = np.where(cm > 0.5, 1, 0)

        pr = predicted.flatten()
        gt = gt.flatten()
        
        tp_img = np.sum(np.logical_and(pr, gt))
        tn_img = np.sum(np.logical_and(np.logical_not(pr), np.logical_not(gt)))
        fp_img = np.sum(np.logical_and(pr, np.logical_not(gt)))
        fn_img = np.sum(np.logical_and(np.logical_not(pr), gt))
        
        
        assert (np.sum([tp_img, fp_img, tn_img, fn_img]) == len(pr))
        
        denominator = tp_img + fn_img + fp_img
        iou = tp_img / denominator if denominator != 0 else 0.0
        
        prediction_made = tp_img + fp_img > 0
        no_object = tp_img + fn_img == 0

        if iou >= IOU_THRESHOLD:
            tp += 1
        elif iou < IOU_THRESHOLD and prediction_made:
            fp += 1
        elif (iou == 0 or not prediction_made) and (not no_object):
            fn += 1
        elif (not prediction_made) and no_object:
            tn += 1


    net_loss = tot_loss / len(dataset.names)
    net_accuracy = 100 * (tp + tn) / (tp + tn + fp + fn + 1e-10)
    prec = tp /(tp + fp) if (tp + fp) != 0 else 0
    rec = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec)  if (prec + rec) != 0 else 0
    

    return {'net_loss': net_loss,
            'net_accuracy': net_accuracy,
            'precision': prec,
            'recall': rec, 
            "f1": f1}

def cluster_image_colors(img, dataset_name, categories):
    ''''
    The image cagteogories are numbers betweeen 0 and 255. We can map them to colors for each category if we 
    map appropriate increasing colors to them.'''
    if dataset_name == 'HRSCD':
        Z = img.reshape((-1,3))
        Z = np.float32(Z)
    else: 
        Z = np.float32(img)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = len(categories)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    out = res.reshape((img.shape))

    return out

def map_to_categorical(img):
    '''
    The categorical mages are based on the indices within them. Also indexes each category.'''
    vals = np.sort(np.unique(img))

    value_to_position = {value: index for index, value in enumerate(vals)}

    positions = np.vectorize(value_to_position.get)(np.ndarray.flatten(img))

    return positions.reshape(img.shape)


def evaluate_img_categorically(y, y_hat, num_changes, y_category, categories, dataset_name, IOU_THRESHOLD=0.5):
    ''''
    Utility funcion to get all images equally evaluated in the categorical loop.
    '''


    out = {c: [0, 0, 0, 0] for c in categories}

    num_changes_predicted = len(cv2.findContours(y_hat.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])
    out['num_changes'] = [num_changes, num_changes_predicted]

    if dataset_name == "HRSCD":
        y_category = y_category[:, :, 0].astype(np.uint8)
    else: 
        y_category = y_category.astype(np.uint8)
    
    y = y.flatten()
    y_hat = y_hat.flatten()
    y_category = y_category.flatten()

    for c in categories:
        if dataset_name == "CSCD" and not ((categories.index(c) + 1) in y_category):
            continue
        
        mask = y_category == categories.index(c) + 1
    
        tp = np.sum((y == 1) & (y_hat == 1) & mask)
        fp = np.sum((y == 0) & (y_hat == 1) & mask)
        tn = np.sum((y == 0) & (y_hat == 0) & mask)
        fn = np.sum((y == 1) & (y_hat == 0) & mask)

        iou = tp / (tp + fn + fp) if (tp + fn + fp) != 0 else 0.0
        prediction_made = np.sum(y_hat) > 0
        no_object = np.sum(y) == 0
        

        if iou >= IOU_THRESHOLD:
            out[c] = [1, 0, 0, 0] #tp
        elif iou < IOU_THRESHOLD and prediction_made:      
            out[c] = [0, 1, 0, 0] #fp
        elif (not prediction_made) and no_object:
            out[c] = [0, 0, 1, 0] #tn
        elif iou == 0 and (not no_object):
            out[c] = [0, 0, 0, 1]  #fn
            
    return out


def evaluate_categories(net, dataset_name, dataset, categories, save_dir, IOU_THRESHOLD=0.5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    '''
    Functional evaluation of all images within the dataset for the given fusion category.
    '''

    categorical_metrics = {}

    for c in categories:
        categorical_metrics[c] = [0, 0, 0, 0] #tp, fp, tn,  fn

    categorical_metrics['num_changes'] = []

    index = 0
    num_changes = 0
    

    for img_index in dataset.names:
        index += 1

        if dataset_name == "CSCD":
            I1, I2, cm, situation, num_changes = dataset.get_img(img_index)
            categorical = np.multiply(cm, categories.index(situation) + 1)
            categorical = np.expand_dims(categorical, axis=0)
        elif dataset_name == "HRSCD" or dataset_name == "HIUCD":
            I1, I2, cm, categorical, num_changes = dataset.get_img(img_index)
            categorical = cluster_image_colors(categorical, dataset_name, categories)
            categorical = map_to_categorical(categorical)
        else:
            I1, I2, cm, _, num_changes = dataset.get_img(img_index)
            print('Not a categorical dataset')
            


        I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
        I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))

        cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).long().to(device))


        output = net(I1, I2).float().to(device)


        predicted = np.exp(np.squeeze(output.cpu().detach().numpy())[1])
        cm = np.squeeze(cm.cpu().detach().numpy())
        cm = np.zeros_like(cm) if np.max(cm) == np.min(cm) else (cm - np.min(cm)) / (np.max(cm) - np.min(cm))
        predicted = np.where(predicted < 0.5, 0, 1)
        
        cm = np.where(cm < 0.5, 0, 1)

        if dataset_name == "CSCD" or dataset_name == "HRSCD" or dataset_name == "HIUCD" :
            curr_metrics = evaluate_img_categorically(cm, predicted, num_changes, categorical, categories, dataset_name)

            for c in categories:
                categorical_metrics[c] = np.add(categorical_metrics[c], curr_metrics[c])
        else: 
            curr_metrics = {'num_changes':[num_changes, 
                                           len(cv2.findContours(predicted.astype(np.uint8),
                                                                cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])]}
            
        categorical_metrics['num_changes'].append(curr_metrics['num_changes'])
        
    b_values = np.array([item[1] for item in categorical_metrics['num_changes']])

    median= np.median(b_values)
    mad = median_abs_deviation(b_values)
    

    filtered_num_changes = [item if abs(item[1] - median) <= 3 * mad else [item[0], 0] for item in categorical_metrics['num_changes']]
    categorical_metrics['num_changes'] = filtered_num_changes
    

    return categorical_metrics


