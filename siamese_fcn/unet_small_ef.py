# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d

'''
The implementation of the Siam_Unet_Early segmentation network, also referred to as FC-EF or Early in the paper/s. 
All creadit to Rodrigo Caye Daudt,  https://rcdaudt.github.io/, Daudt, R. C., Le Saux, B., & Boulch, A. 
"Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
'''


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(p=0.2),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)

class Small_UNet_EF(nn.Module):


    def __init__(self, input_channels, out_channels):
        super(Small_UNet_EF, self).__init__()

        self.input_channels = input_channels
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        features = [8, 16, 32]
        for feature in features:
            self.downs.append(DoubleConv(input_channels, feature))
            input_channels = feature
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):
        x = torch.cat((x1, x2), 1)

        skip_connections = []
        for down in self.downs:
            x = down(x).to(x.device)
            skip_connections.append(x)
            x = self.pool(x)
        # max pools will floor inputs unless divisible by 16 - we can make the implementation general

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True) # as it will always be smaller


            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return self.sm(x)

    
