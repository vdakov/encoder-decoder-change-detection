import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from unet_small_ef import DoubleConv

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d


'''
The implementation of the SiamUnet_conc segmentation network, also referred to as Middle-Conc or FC-Siam-Conc in the paper/s. 
All creadit to Rodrigo Caye Daudt,  https://rcdaudt.github.io/, Daudt, R. C., Le Saux, B., & Boulch, A. 
"Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.
'''

class SiamUNet_Middle_Small(nn.Module):


    def __init__(self, input_channels, out_channels):
        super(SiamUNet_Middle_Small, self).__init__()

        self.input_channels = input_channels
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        features = [8, 16, 32]
        for feature in features:
            self.downs.append(DoubleConv(input_channels, feature))
            input_channels = feature
        
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups.append(DoubleConv(feature*3, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.sm = nn.LogSoftmax(dim=1)
    def forward(self, x1, x2):
        skip_connections_x1 = []
        skip_connections_x2 = []

        for down in self.downs:
            x1 = down(x1) 
            x2 = down(x2)
            skip_connections_x1.append(x1)
            skip_connections_x2.append(x2)
            x1 = self.pool(x1)
            x2 = self.pool(x2)
        # max pools will floor inputs unless divisible by 16 - we can make the implementation general

        x = self.bottleneck(x1)
        skip_connections_x1 = skip_connections_x1[::-1]
        skip_connections_x2 = skip_connections_x2[::-1] 

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            x1_res = skip_connections_x1[idx//2]
            x2_res = skip_connections_x2[idx//2]

            # if x.shape != torch.cat((x1_res, x2_res, x)).shape:
            #     x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True) # as it will always be smaller

            concat_skip = torch.cat((x1_res, x2_res, x), axis=1)
            x = self.ups[idx + 1](concat_skip)

        x = self.final_conv(x)

        return self.sm(x)
    
