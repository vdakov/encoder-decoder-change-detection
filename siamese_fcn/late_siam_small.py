# Rodrigo Caye Daudt
# https://rcdaudt.github.io/
# Daudt, R. C., Le Saux, B., & Boulch, A. "Fully convolutional siamese networks for change detection". In 2018 25th IEEE International Conference on Image Processing (ICIP) (pp. 4063-4067). IEEE.

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.padding import ReplicationPad2d
from unet_small_ef import DoubleConv

class Small_UNet_Late(nn.Module):


    def __init__(self, input_channels, out_channels):
        super(Small_UNet_Late, self).__init__()

        self.input_channels = input_channels
        input_channels_init = int(input_channels / 2)
        self.input_channels_init = input_channels_init
        self.out_channels = out_channels
        input_channels_late = out_channels * 2
        self.input_channels_late = out_channels * 2
        self.ups_init = nn.ModuleList()
        self.downs_init = nn.ModuleList()
        self.ups_late = nn.ModuleList()
        self.downs_late = nn.ModuleList()

        features = [8, 16, 32]
        for feature in features:
            self.downs_init.append(DoubleConv(input_channels_init, feature))
            self.downs_late.append(DoubleConv(input_channels_late, feature))
            input_channels_init = feature
            input_channels_late = feature
        
        for feature in reversed(features):
            self.ups_init.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups_late.append(nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2))
            self.ups_init.append(DoubleConv(feature*2, feature))
            self.ups_late.append(DoubleConv(feature*4, feature))

        self.bottleneck_init = DoubleConv(features[-1], features[-1] * 2)
        self.bottleneck_late = DoubleConv(features[-1], features[-1] * 2)
        self.final_conv_init = nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.final_conv_late= nn.Conv2d(features[0], out_channels, kernel_size=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)


        self.sm = nn.LogSoftmax(dim=1)

    def forward(self, x1, x2):

        skip_connections_x1 = []
        skip_connections_x2 = []

        for down in self.downs_init:
            x1 = down(x1)
            skip_connections_x1.append(x1)
            x1 = self.pool(x1)
        
        for down in self.downs_init:
            x2 = down(x2)
            skip_connections_x2.append(x2)
            x2 = self.pool(x2)
        # max pools will floor inputs unless divisible by 16 - we can make the implementation general

        x1 = self.bottleneck_init(x1)
        x2 = self.bottleneck_init(x2)

        skip_connections_x1 = skip_connections_x1[::-1]
        skip_connections_x2 = skip_connections_x2[::-1]

        for idx in range(0, len(self.ups_init), 2):
            x1 = self.ups_init[idx](x1)
            x2 = self.ups_init[idx](x2)
            skip_connection_x1 = skip_connections_x1[idx//2]
            skip_connection_x2 = skip_connections_x2[idx//2]

            if x1.shape != skip_connection_x1.shape:
                x1 = F.interpolate(x1, size=skip_connection_x1.shape[2:], mode="bilinear", align_corners=True) # as it will always be smaller
            if x2.shape != skip_connection_x2.shape:
                x2 = F.interpolate(x2, size=skip_connection_x2.shape[2:], mode="bilinear", align_corners=True) # as it will always be smaller


            concat_skip_x1 = torch.cat((skip_connection_x1, x1), dim=1)
            concat_skip_x2 = torch.cat((skip_connection_x2, x2), dim=1)
            x1 = self.ups_init[idx + 1](concat_skip_x1)
            x2 = self.ups_init[idx + 1](concat_skip_x2)

        x1 = self.final_conv_init(x1)
        x2 = self.final_conv_init(x2)

        x = torch.cat((x1, x2), 1)

        skip_connections = []
        for down in self.downs_late:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)
        # max pools will floor inputs unless divisible by 16 - we can make the implementation general

        x = self.bottleneck_late(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups_late), 2):
            x = self.ups_late[idx](x)
            skip_connection = skip_connections[idx//2]
            skip_connection_x1 = skip_connections_x1[idx//2]
            skip_connection_x2 = skip_connections_x2[idx//2]

            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode="bilinear", align_corners=True) # as it will always be smaller


            concat_skip = torch.cat((skip_connection, x, skip_connection_x1, skip_connection_x2), dim=1)
            x = self.ups_late[idx + 1](concat_skip)

        x = self.final_conv_late(x)


        return self.sm(x)

    
