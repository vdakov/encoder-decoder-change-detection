import copy
import torch
from torch.autograd import Variable
from metrics import evaluate_net_predictions
from tqdm import tqdm as tqdm
import time

def train(net, train_loader, val_loader, criterion, device, n_epochs = 10, save = True, save_dir=f'{time.time()}.pth.tar', skip_val = True, early_stopping = True):
    '''
    The current training and validation loop used for the experiment. Here all major hyperparameters and machine learning 
    methods have been employed. They include: 
        - Adam Optimizer
        - Exponential Learning Rate Scheduler 
        - Early Stopping 
        - Validation Sets (and an optional stopping of it desired for testing/computation purposes)
        
    The parameters for Adam and the exponential scheduler are hardcoded and borrowed from https://ieeexplore.ieee.org/abstract/document/8451652.
    
    The patience paramter is hardcoded to 10 epochs. When patience is restored it is not put to 10, but to 5 (unless already bigger).
    
    Training is performed in batches using PyTorch's dataloaders. 
    '''
    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    
    train_metrics = []
    val_metrics = []
    
    patience = 10
    best_loss = float('inf')
    best_model_weights = copy.deepcopy(net.state_dict())
    
    for epoch in range(n_epochs):
        net.train()
        
        with tqdm(total=len(train_loader), desc=f'Epoch {epoch+1}/{n_epochs}', unit='batch') as pbar:
            for batch in train_loader:
                I1 = Variable(batch['I1'].float().to(device))
                I2 = Variable(batch['I2'].float().to(device))
                label = Variable(batch['label'].long().to(device))
        
                optimizer.zero_grad()

                output = net(I1, I2).to(device)
                loss = criterion(output, label)
                
                loss.backward()
                optimizer.step()

                pbar.set_postfix(loss=loss.item())
                pbar.update(1)

        scheduler.step()


        train_metrics.append(evaluate_net_predictions(net, criterion, val_loader))
        if not skip_val:
            curr_val_metrics = evaluate_net_predictions(net, criterion, val_loader)
            val_metrics.append(curr_val_metrics)

            val_loss = curr_val_metrics['net_loss']
            
            if val_loss < best_loss:
                best_loss = val_loss
                best_model_weights = copy.deepcopy(net.state_dict())  
                if early_stopping:
                    patience = max(5, patience)
            else:
                if early_stopping and epoch > 10:
                    patience -= 1
                if patience == 0:
                    break
    if save:
        if early_stopping:
            torch.save(best_model_weights, save_dir)
        else: 
            torch.save(net.state_dict(), save_dir)


    return train_metrics, val_metrics