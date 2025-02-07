import csv
import os
from matplotlib import pyplot as plt
import torch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
import numpy as np

def evaluate_net_predictions(net, criterion, dataset_loader):
    '''A function that looks at the net loss, and accuracy metrics for a specifc. It evaluates all images per 
    batch, which is why the IoU and reshaping computations are a bit weirder than they seem. The IOU threshold is fixed in its own function 
    (as of time of writing =0.5).
    
    It is used to monitor how both the loss and accuracy metrics are doing during training.
    '''
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    tot_loss = 0
    
    for batch in dataset_loader:
        I1_batch = Variable(batch['I1'].float().to(device))
        I2_batch = Variable(batch['I2'].float().to(device))
        cm_batch = Variable(batch['label'].long().to(device))


        output_batch = net(I1_batch, I2_batch).to(device)
        loss = criterion(output_batch, cm_batch)
        tot_loss += loss.item()
        
        predicted_labels  = torch.exp(output_batch[:, 1, :, :])
        predicted_labels = (predicted_labels > 0.5).long()
            
        cm_batch_min = cm_batch.min(dim=1, keepdim=True)[0]
        cm_batch_max = cm_batch.max(dim=1, keepdim=True)[0]
        cm_batch_normalized = (cm_batch - cm_batch_min) / (cm_batch_max - cm_batch_min + 1e-6)
        gt_labels = (cm_batch_normalized > 0.5).long()
        
        #flatten the batches, but without the actual batch
        pr = predicted_labels.view(predicted_labels.size(0), -1)
        gt = gt_labels.view(gt_labels.size(0), -1)

  
        # # Calculate TP, TN, FP, FN for the entire batch
        iou_batch = compute_iou_batch(pr, gt)
        
        tp += iou_batch[0]
        fp += iou_batch[1]
        tn += iou_batch[2]
        fn += iou_batch[3]
        
    net_loss = tot_loss / len(dataset_loader.dataset)
    net_accuracy = 100 * (tp + tn) / (tp + tn + fp + fn + 1e-10)
    prec = tp /(tp + fp) if (tp + fp) != 0 else 0
    rec = tp / (tp + fn) if (tp + fn) != 0 else 0
    f1 = 2 * (prec * rec) / (prec + rec)  if (prec + rec) != 0 else 0
    

    return {'net_loss': net_loss,
            'net_accuracy': net_accuracy,
            'precision': prec,
            'recall': rec, 
            "f1": f1}
    

def compute_iou_batch(predictions, ground_truth, IOU_THRESHOLD=0.5):
    '''The IoU TP, FP, TN and FN evaluator per image. It receives as input batches, it treats as 
    entire tensors. 
    
    Despite the fact TN is non-traditional in object detection, here the actual task is a bit different. We evaluate 
    on a pixel level, meaning it is a binary classification task, where this method of classification is fully valid. 
    
    The threshold of 0.5 has been selected as industry standard. 
    '''

    intersection = (predictions * ground_truth).sum(dim=1)
    union = predictions.sum(dim=1) + ground_truth.sum(dim=1) - intersection
    #should be in shapes (batch size, intersection), (batch size, union)
    
    iou = intersection / (union + 1e-6)
    
    tp = (iou >= IOU_THRESHOLD).sum().item()
    fp = ((iou < IOU_THRESHOLD) & (intersection > 0)).sum().item()
    tn = ((union == 0) & (intersection == 0)).sum().item()
    fn = ((union > 0) & (intersection == 0)).sum().item()
    return [tp, fp, tn, fn]

def get_ground_truth(dataset):
    ''' A getter for the ground truth dataset in-memory. Outputs a normal list
    '''
    
    ground_truth = []
    for img_index in dataset.names:
        _, _, cm, _= dataset.get_img(img_index)
        cm = np.zeros_like(cm) if np.max(cm) == np.min(cm) else (cm - np.min(cm)) / (np.max(cm) - np.min(cm))
        cm = np.where(cm < 0.5, 0, 1)
        ground_truth.append(cm)
                
    return ground_truth 

def get_predictions(net, dataset):
    ''' A getter for the predictions of all architectures in-memory. Outputs a list that is later used in a dictionary.
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)
    predictions = []
    
    for img_index in dataset.names:
        I1, I2, _, _= dataset.get_img(img_index)
        I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
        I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))
        
        output = net(I1, I2).float().to(device)

        predicted = np.exp(np.squeeze(output.cpu().detach().numpy())[1])
        predicted = np.where(predicted < 0.5, 0, 1)

        predictions.append(predicted)
                
    return predictions 





