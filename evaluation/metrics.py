import torch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
import numpy as np
from math import ceil

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
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))
    tot_loss = 0
    tot_count = 0

    for img_index in dataset.names:
        img1, img2, label = dataset.get_img(img_index)

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
