import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from torch.utils.data import DataLoader, SubsetRandomSampler
import os
import torch
from torch.autograd import Variable
import numpy as np


def examine_subset(net, model_name, dataset, num_samples, device, save_path=None):

    p, q = num_samples // 4, 4

    fig = plt.figure(figsize=(10, q * 2))
    outer = gridspec.GridSpec(num_samples // 4, 4)
    fig.suptitle(model_name, fontsize=16)

    indices = torch.randperm(len(dataset)).tolist()[:num_samples]
    sampler = SubsetRandomSampler(indices)

    batch_size = 1
    data_loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    k = 0
    for i in range(p):
        for j in range(q):
            batch = next(iter(data_loader))

            img1 = batch['I1']
            img2 = batch['I2']
            label = batch['label']
            I1 = Variable(batch['I1'].float().to(device))
            I2 = Variable(batch['I2'].float().to(device))
            label = Variable(batch['label'].float().to(device))

            output = net(I1, I2)

            img1 = np.transpose(np.squeeze(img1.cpu().numpy()), (1, 2, 0))
            img2 = np.transpose(np.squeeze(img2.cpu().numpy()), (1, 2, 0))
            label = np.squeeze(label.cpu().numpy())
            output = np.exp(np.squeeze(output.cpu().detach().numpy())[1])
            output = np.where(output < 0.5, 0, 1)


            inner = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=outer[i, j], wspace=0.1, hspace=0.1)

            ax1 = plt.Subplot(fig, inner[0, 0])
            ax1.imshow(img1, cmap='gray')
            ax1.set_title('T1')
            ax1.axis('off')
            fig.add_subplot(ax1)

            ax2 = plt.Subplot(fig, inner[0, 1])
            ax2.imshow(img2, cmap='gray')
            ax2.set_title('T2')
            ax2.axis('off')
            fig.add_subplot(ax2)

            ax3 = plt.Subplot(fig, inner[1, 0])
            ax3.imshow(label, cmap='gray')
            ax3.set_title('GT')
            ax3.axis('off')
            fig.add_subplot(ax3)

            ax4 = plt.Subplot(fig, inner[1, 1])
            ax4.imshow(output, cmap='gray')
            ax4.set_title('OUT')
            ax4.axis('off')
            fig.add_subplot(ax4)

    if save_path:
        plt.savefig(os.path.join(save_path, 'data_examination.png'))