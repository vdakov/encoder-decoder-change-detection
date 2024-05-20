import torch
from torch.autograd import Variable
from metrics import evaluate_net_predictions
from tqdm import tqdm as tqdm
import time



def train(net, net_name, train_dataset, train_loader, val_dataset, criterion, patch_size, device, n_epochs = 10, save = True, save_dir=f'{time.time()}.pth.tar'):

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    epoch_metrics = {"train": [], "val": []}

    for _ in range(n_epochs):
        net.train()

        for batch in tqdm(train_loader):
            I1 = Variable(batch['I1'].float().to(device))
            I2 = Variable(batch['I2'].float().to(device))
            label = Variable(batch['label'].float().to(device))

            
            optimizer.zero_grad()
            output = net(I1, I2).to(device)

            output_rounded = output
            label_rounded = (label / 255).long().to(device)  # Assuming label values are in range [0, 255]

            loss = criterion(output_rounded, label_rounded)


            loss.backward()
            optimizer.step()

        scheduler.step()

        epoch_metrics['train'].append(evaluate_net_predictions(net, criterion, train_dataset, patch_size))
        epoch_metrics['val'].append(evaluate_net_predictions(net, criterion, val_dataset, patch_size))


    if save:

        torch.save(net.state_dict(), save_dir)


    return epoch_metrics
