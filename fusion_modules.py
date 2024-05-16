import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class EarlyFusionNet(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EarlyFusionNet, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x1, x2):
        x = torch.cat((x1, x2)) #look into whether this is the dimension to be

        x_enc = self.encoder.forward(x)
        x_dec = self.decoder.forward(x_enc)

        return x_dec
    
class MiddleFusionNet(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EarlyFusionNet, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x1, x2):
        x1_enc = self.encoder.forward(x1) 
        x2_enc = self.encoder.forward(x2)

        x_enc = self.fuse(x1_enc, x2_enc)
        x_dec = self.decoder.forward(x_enc)

        return x_dec
    
    def fuse(x1, x2):
        pass 

class LateFusionNet(nn.Module):
    def __init__(self, encoder, decoder, **kwargs):
        super(EarlyFusionNet, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, x1, x2):

        x1_enc = self.encoder.forward(x1) 
        x2_enc = self.encoder.forward(x2)

        x1_dec = self.decoder.forward(x1_enc)
        x2_dec = self.decoder.forward(x2_enc)

        return self.fuse(x1_dec, x2_dec)
    
    def fuse(x1, x2):
        pass 
    

