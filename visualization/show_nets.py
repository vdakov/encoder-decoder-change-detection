from torchviz import make_dot
import os 
import torch
from torch.autograd import Variable


def show_network(net, model_name, dataloader, device):
    batch = next(iter(dataloader))

    I1 = batch['I1']
    I2 = batch['I2']
    label = batch['label']

    I1 = Variable(I1.float().to(device))
    I2 = Variable(I1.float().to(device))
    label = Variable(label.float().to(device))

    yhat = torch.round(net(I1, I2))
    make_dot(yhat, params=dict(list(net.named_parameters()))).render(os.path.join('results', 'models', f'{model_name}.png'), format="png")