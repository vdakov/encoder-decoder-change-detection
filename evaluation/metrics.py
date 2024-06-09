import torch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
import numpy as np
import cv2

IOU_THRESHOLD = 0.5


def evaluate_net_prediction_batch(predictions, ground_truths, IOU_THRESHOLD=0.5):
    predictions = predictions.cpu().detach().numpy()
    ground_truths = ground_truths.cpu().detach().numpy()
    
    batch_size = predictions.shape[0]
    
    results = [0, 0, 0, 0]

    for i in range(batch_size):
        prediction = predictions[i]
        ground_truth = ground_truths[i]
        
        predicted =  np.exp(np.squeeze(prediction)[1])
        predicted = np.where(predicted < 0.5, 0, 1)
        
        ground_truth = np.squeeze(ground_truth)
        ground_truth = (ground_truth - np.min(ground_truth)) / (np.ptp(ground_truth)) if np.ptp(ground_truth) != 0 else np.zeros_like(ground_truth)
        
        pr = np.where(predicted > 0.5, 1, 0)
        gt = np.where(ground_truth > 0.5, 1, 0)
        
        pr = pr.flatten()
        gt = gt.flatten()
        
        tp_img = np.sum(pr & gt)
        tn_img = np.sum(~pr & ~gt)
        fp_img = np.sum(pr & ~gt)
        fn_img = np.sum(~pr & gt)
        
        iou = tp_img / max(tp_img + fn_img + fp_img, 1e-10)
        no_object = tp_img + fn_img == 0

        result = None
        
        if iou >= min(IOU_THRESHOLD, 1):
            result = [1, 0, 0, 0] #tp
        elif iou > 0 and no_object:
            results = [0, 1, 0, 0] #fp
        elif iou < IOU_THRESHOLD and no_object:
            result = [0, 0, 1, 0] #tn
        elif iou < IOU_THRESHOLD and (not no_object):
            result = [0, 0, 0, 1] #fn

        else:
            raise ValueError('You shoudn\'t be here')
        
        results = np.add(results, result)


    
    batch_accuracy = 100 * (results[0] + results[2]) / batch_size
    prec = results[0] / max(1, (results[0] + results[1]))
    rec = results[0] / max(1, (results[0] + results[3]))


    return {'batch_accuracy': batch_accuracy, 
            'precision': prec, 
            'recall': rec}
    

def evaluate_net_predictions(net, criterion, dataset):
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    tp = 0 
    tn = 0
    fp = 0 
    fn = 0 

    tot_loss = 0
    tot_count = 0

    for img_index in dataset.names:
        I1, I2, cm, _, _ = dataset.get_img(img_index)

        I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
        I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))
        cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).long().to(device))

        output = net(I1, I2).float().to(device)
                
        loss = criterion(output, cm)
        tot_loss += loss.data * np.prod(cm.size())
        tot_count += np.prod(cm.size())

        predicted =  np.exp(np.squeeze(output.cpu().detach().numpy())[1])
        predicted = np.where(predicted < 0.5, 0, 1)

        cm = np.squeeze(cm.cpu().detach().numpy())
        cm = (cm - np.min(cm)) / (np.ptp(cm)) if np.ptp(cm) != 0 else np.zeros_like(cm)
        gt = np.where(cm > 0.5, 1, 0)

        pr = predicted.flatten()
        gt = gt.flatten()


                
        tp_img = np.sum(pr & gt)
        tn_img = np.sum(~pr & ~gt)
        fp_img = np.sum(pr & ~gt)
        fn_img = np.sum(~pr & gt)

        iou = tp_img / max(tp_img + fn_img + fp_img, 1e-10)
        # print(iou)
        no_object = tp_img + fn_img == 0

        if iou >= min(IOU_THRESHOLD, 1):
            tp += 1
        if iou > 0 and no_object: 
            fp += 1
        if iou < IOU_THRESHOLD and (not no_object):
            fn += 1
        if iou < IOU_THRESHOLD and no_object:
            tn += 1
        

    net_loss = tot_loss/tot_count        
    net_loss = float(net_loss.cpu().numpy())
    
    net_accuracy = 100 * (tp + tn)/tot_count
    

    prec = tp / max(1, (tp + fp))
    rec = tp / max(1, (tp + fn))


    return {'net_loss': net_loss, 
            'net_accuracy': net_accuracy, 
            'precision': prec, 
            'recall': rec}

def cluster_image_colors(img, categories):
    Z = img.reshape((-1,3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = len(categories)
    _, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    out = res.reshape((img.shape))

    return out

def map_to_categorical(img):
    vals = np.sort(np.unique(img))

    value_to_position = {value: index for index, value in enumerate(vals)}
    
    # Assumes img is one dimensional
    positions = np.vectorize(value_to_position.get)(np.ndarray.flatten(img))
    
    return positions.reshape(img.shape)


def evaluate_img_categorically(y, y_hat, num_changes, y_category, categories):


    out = {c: [0, 0, 0, 0] for c in categories}
    
    
    num_changes_predicted = len(cv2.findContours(cv2.cvtColor(y_hat.copy(), cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])
    out['num_changes'] = [num_changes, num_changes_predicted]
    
    y_category = y_category[:, :, 0].astype(int)
    
    for c in categories:
        
        mask = y_category == categories.index(c)  # Create a mask for the current category


        tp = np.sum((y == 1) & (y_hat == 1) & mask)
        fp = np.sum((y == 1) & (y_hat == 0) & mask)
        tn = np.sum((y == 0) & (y_hat == 0) & mask)
        fn = np.sum((y == 0) & (y_hat == 1) & mask)

        iou = tp / max(tp + fn + fp, 1e-10)
        no_object = tp + fn == 0

        if iou >= min(IOU_THRESHOLD, 1):
            out[c] = [1, 0, 0, 0] #tp
        if iou > 0 and no_object: 
            out[c] = [0, 1, 0, 0] #fp
        if iou < IOU_THRESHOLD and no_object:
            out[c] = [0, 0, 1, 0] #tn 
        if iou < IOU_THRESHOLD and (not no_object):
            out[c] = [0, 0, 0, 1]  #fn
        

    return out


def evaluate_categories(net, dataset_name, dataset, categories):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    categorical_metrics = {}

    for c in categories:
        categorical_metrics[c] = [0, 0, 0, 0] #tp, fp, tn,  fn
        
    categorical_metrics['num_changes'] = []

    index = 0
    
    num_changes = 0

    for img_index in dataset.names:
        index += 1

        if dataset_name is "CSCD":
            I1, I2, cm, situation, num_changes = dataset.get_img(img_index)
            # categorical = np.divide(cv2.cvtColor(cm, cv2.COLOR_GRAY2RGB), 255)
            categorical = np.multiply(cm, categories.index(situation))
            categorical = np.expand_dims(categorical, axis=0)
            print(categorical.shape)


        elif dataset_name in ["HRSCD", "HIUCD"]:
            I1, I2, cm, categorical = dataset.get_img(img_index)
            num_changes = len(cv2.findContours(cv2.cvtColor(cm.copy(), cv2.COLOR_BGR2GRAY), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0])
            categorical = cluster_image_colors(categorical, categories)

            categorical = map_to_categorical(categorical)
        else:
            print('Not a categorical dataset')
            break


        I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
        I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))

        cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).long().to(device))


        output = net(I1, I2).float().to(device)

        _, predicted = torch.max(output.data, 1)


        predicted = np.exp(np.squeeze(output.cpu().detach().numpy())[1])
        cm = np.squeeze(cm.cpu().detach().numpy())
        cm = (cm - np.min(cm)) / (np.max(cm) - np.min(cm))
        predicted = np.where(predicted < 0.5, 0, 1)
        cm = np.where(cm < 0.5, 0, 1)

        curr_metrics = evaluate_img_categorically(cm, predicted, num_changes, categorical, categories)

        for c in categories:
            categorical_metrics[c] = np.add(categorical_metrics[c], curr_metrics[c])
        categorical_metrics['num_changes'].append(curr_metrics['num_changes'])



    return categorical_metrics


