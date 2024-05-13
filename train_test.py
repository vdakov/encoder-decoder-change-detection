# PyTorch
import torch
from torch.autograd import Variable
from metrics import evaluate_net_predictions
from tqdm import tqdm as tqdm
import time

def train(net, net_name, train_dataset, train_loader, val_dataset, criterion, n_epochs = 10, save = True, save_dir=f'{time.time()}.pth.tar'):

    optimizer = torch.optim.Adam(net.parameters(), weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.95)
    epoch_metrics = {"train": [], "val": []}
    
    for _ in range(n_epochs):
        net.train()

        for batch in train_loader:
            I1 = Variable(batch['I1'].float().cuda())
            I2 = Variable(batch['I2'].float().cuda())
            label = torch.squeeze(Variable(batch['label'].cuda()))
       
            optimizer.zero_grad()
            output = torch.squeeze(net(I1, I2))
            visualize_inputs_and_predictions(I1, I2, output)


            loss = criterion(torch.flatten(output), torch.flatten(label.long()))

            print(loss)
                
            loss.backward()
            optimizer.step()
            
        scheduler.step()

        epoch_metrics['train'].append(evaluate_net_predictions(net, criterion, train_dataset))
        epoch_metrics['val'].append(evaluate_net_predictions(net, criterion, val_dataset))


    if save:

        torch.save(net.state_dict(), save_dir)
        

    return epoch_metrics





def visualize_inputs_and_predictions(I1, I2, output):
    # Convert tensors to numpy arrays
    I1_np = I1.cpu().detach().numpy()
    I2_np = I2.cpu().detach().numpy()
    output_np = output.cpu().detach().numpy()
    
    # Plot I1, I2, and output
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    axs[0].imshow(I1_np)
    axs[0].set_title('Input I1')
    axs[1].imshow(I2_np.transpose(1, 2, 0))
    axs[1].set_title('Input I2')
    axs[2].imshow(output_np, cmap='gray')
    axs[2].set_title('Predicted Output')
    plt.show()