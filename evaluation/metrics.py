import torch
from torch.autograd import Variable
from tqdm import tqdm as tqdm
import numpy as np
from math import ceil

L = 1024

def evaluate_net_predictions(net, criterion, dataset):
    net.eval()
    tp = 0 
    tn = 0
    fp = 0 
    fn = 0 

    n = 2
    class_correct = list(0. for i in range(n))
    class_total = list(0. for i in range(n))
    class_accuracy = list(0. for i in range(n))

    for img_index in dataset.names:
        img1, img2, label = dataset.get_img(img_index)

        s = label.shape

    for ii in range(ceil(s[0] / L)) :
        for jj in range(ceil(s[1] / L)):
            xmin = L*ii
            xmax = min(L*(ii+1),s[1])
            ymin = L*jj
            ymax = min(L*(jj+1),s[1])

            I1 = img1[:, xmin:xmax, ymin:ymax]
            I2 = img2[:, xmin:xmax, ymin:ymax]
            cm = label[xmin:xmax, ymin:ymax]

            # I1 = Variable(torch.unsqueeze(I1, 0).float()).cuda()
            # I2 = Variable(torch.unsqueeze(I2, 0).float()).cuda()
            # cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).float()).cuda()
            I1 = Variable(torch.unsqueeze(I1, 0).float())
            I2 = Variable(torch.unsqueeze(I2, 0).float())
            cm = Variable(torch.unsqueeze(torch.from_numpy(1.0*cm),0).float())

            output = net(I1, I2)
                    
            loss = criterion(output, cm.long())
            tot_loss += loss.data * np.prod(cm.size())
            tot_count += np.prod(cm.size())

            _, predicted = torch.max(output.data, 1)

            c = (predicted.int() == cm.data.int())
            for i in range(c.size(1)):
                for j in range(c.size(2)):
                    l = int(cm.data[0, i, j])
                    class_correct[l] += c[0, i, j]
                    class_total[l] += 1
                    
            pr = (predicted.int() > 0).cpu().numpy()
            gt = (cm.data.int() > 0).cpu().numpy()
            
            tp += np.logical_and(pr, gt).sum()
            tn += np.logical_and(np.logical_not(pr), np.logical_not(gt)).sum()
            fp += np.logical_and(pr, np.logical_not(gt)).sum()
            fn += np.logical_and(np.logical_not(pr), gt).sum()

    net_loss = tot_loss/tot_count        
    net_loss = float(net_loss.cpu().numpy())
    
    net_accuracy = 100 * (tp + tn)/tot_count
    
    for i in range(n):
        class_accuracy[i] = 100 * class_correct[i] / max(class_total[i],0.00001)
        class_accuracy[i] =  float(class_accuracy[i].cpu().numpy())

    prec = tp / (tp + fp)
    rec = tp / (tp + fn)


    return {'net_loss': net_loss, 
            'net_accuracy': net_accuracy, 
            'class_accuracy': class_accuracy, 
            'precision': prec, 
            'recall': rec}
