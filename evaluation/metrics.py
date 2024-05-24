import torch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
import numpy as np
from math import ceil
import cv2

L = 1024

def evaluate_net_predictions(net, criterion, dataset, patch_size):
    net.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    tp = 0 
    tn = 0
    fp = 0 
    fn = 0 

    n = 2

    tot_loss = 0
    tot_count = 0

    for img_index in dataset.names:
        img1, img2, label, _ = dataset.get_img(img_index)

        s = label.shape

        for ii in range(ceil(patch_size / L)) :
            for jj in range(ceil(patch_size / L)):


                xmin = L*ii
                xmax = min(L*(ii+1),s[1])
                ymin = L*jj
                ymax = min(L*(jj+1),s[1])

                I1 = img1[:, xmin:xmax, ymin:ymax]
                I2 = img2[:, xmin:xmax, ymin:ymax]
                cm = label[xmin:xmax, ymin:ymax]
                cm = cm.astype(float) / 255
                

    
                I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
                I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).long().to(device))

                output = net(I1, I2).float().to(device)
                        
                loss = criterion(output, cm)
                tot_loss += loss.data * np.prod(cm.size())
                tot_count += np.prod(cm.size())

                _, predicted = torch.max(output.data, 1)


                predicted = np.squeeze(output.cpu().detach().numpy())[0] -np.squeeze(output.cpu().detach().numpy())[1]
                predicted = (predicted - np.min(predicted)) / (np.max(predicted) - np.min(predicted))
                cm = np.squeeze(cm.cpu().detach().numpy())
                cm = (cm - np.min(cm)) / (np.max(cm) - np.min(cm))
                predicted = np.where(predicted < 0.5, 0, 1)
                cm = np.where(cm < 0.5, 0, 1)
                
                pr = np.where(predicted > 0.5 , 1, 0)
                gt = np.where(cm > 0.5, 1, 0)
                        
                # pr = (predicted.int() > 0).cpu().numpy()
                # gt = (cm.data.int() > 0).cpu().numpy()
                
                tp += np.logical_and(pr, gt).sum()
                tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
                fp += np.logical_and(pr, np.logical_not(gt)).sum()
                fn += np.logical_and(np.logical_not(pr), gt).sum()

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
    ret, label, center=cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
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


def evaluate_img_categorically(y, y_hat, y_category, categories):


    out = {c: [0, 0, 0, 0] for c in categories}
    y_category = y_category[:, :, 0].astype(int)
    

    for c in categories:
        mask = y_category == categories.index(c)  # Create a mask for the current category

        tp = np.sum((y == 1) & (y_hat == 1) & mask)
        fp = np.sum((y == 1) & (y_hat == 0) & mask)
        tn = np.sum((y == 0) & (y_hat == 0) & mask)
        fn = np.sum((y == 0) & (y_hat == 1) & mask)
        
        out[c] = [tp, fp, tn, fn]


    return out



def evaluate_categories(net, dataset, categories, patch_size):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    categorical_metrics = {}

    for c in categories:
        categorical_metrics[c] = [0, 0, 0, 0] #tp, fp, tn,  fn

    index = 0

    for img_index in dataset.names:
        index += 1
        img1, img2, label, categorical = dataset.get_img(img_index)
        categorical = cluster_image_colors(categorical, categories)
        categorical = map_to_categorical(categorical)


        s = label.shape

        for ii in range(ceil(patch_size / L)) :
            for jj in range(ceil(patch_size / L)):


                xmin = L*ii
                xmax = min(L*(ii+1),s[1])
                ymin = L*jj
                ymax = min(L*(jj+1),s[1])

                I1 = img1[:, xmin:xmax, ymin:ymax]
                I2 = img2[:, xmin:xmax, ymin:ymax]
                cm = label[xmin:xmax, ymin:ymax]
                categ_img = categorical[xmin:xmax, ymin:ymax]
                cm = cm.astype(float) / 255
                

                I1 = Variable(torch.unsqueeze(I1, 0).float().to(device))
                I2 = Variable(torch.unsqueeze(I2, 0).float().to(device))
                cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).long().to(device))


                output = net(I1, I2).float().to(device)

                _, predicted = torch.max(output.data, 1)


                predicted = np.squeeze(output.cpu().detach().numpy())[0] -np.squeeze(output.cpu().detach().numpy())[1]
                predicted = (predicted - np.min(predicted)) / (np.max(predicted) - np.min(predicted))
                cm = np.squeeze(cm.cpu().detach().numpy())
                cm = (cm - np.min(cm)) / (np.max(cm) - np.min(cm))
                predicted = np.where(predicted < 0.5, 0, 1)
                cm = np.where(cm < 0.5, 0, 1)
                
                curr_metrics = evaluate_img_categorically(label, predicted, categorical, categories)

                for c in categories:
                    categorical_metrics[c] = np.add(categorical_metrics[c], curr_metrics[c])
                        
        

    return categorical_metrics

