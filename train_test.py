import torch
from torch.autograd import Variable
from metrics import evaluate_net_predictions
from metrics import evaluate_net_prediction_batch
from tqdm import tqdm as tqdm
import time
import numpy as np

def train(net, train_dataset, train_loader, val_dataset, criterion, device, n_epochs = 10, save = True, save_dir=f'{time.time()}.pth.tar', skip_val = True, early_stopping = True):

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    train_metrics = []
    val_metrics = []
    prev_val_metrics = {'net_loss':0, 'net_accuracy':0, 'precision':0, 'recall':0}
    n_batches = 0

    for epoch in range(n_epochs):
        net.train()

        cumulative_metrics = {metric: 0 for metric in ['net_loss', 'net_accuracy', 'precision', 'recall']}

        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                I1 = Variable(batch['I1'].float().to(device))
                I2 = Variable(batch['I2'].float().to(device))
                label = Variable(batch['label'].long().to(device))

                optimizer.zero_grad()

                output = net(I1, I2).to(device)


                loss = criterion(output, label)
                batch_metrics = evaluate_net_prediction_batch(output, label)

                cumulative_metrics['net_loss'] += loss.item()
                cumulative_metrics['net_accuracy'] += batch_metrics['batch_accuracy']
                cumulative_metrics['precision'] += batch_metrics['precision']
                cumulative_metrics['recall'] += batch_metrics['recall']
                
                n_batches += 1
                loss.backward()
                optimizer.step()

                
                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        scheduler.step()

        train_metrics_epoch = {}
        for metric in cumulative_metrics:
          train_metrics_epoch[metric] = cumulative_metrics[metric] / n_batches
        
        train_metrics.append(train_metrics_epoch)
        if not skip_val:
            curr_val_metrics = evaluate_net_predictions(net, criterion, val_dataset)
            val_metrics.append(curr_val_metrics)


    if save:

        torch.save(net.state_dict(), save_dir)


    return train_metrics, val_metrics
