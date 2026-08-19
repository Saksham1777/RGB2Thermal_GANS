import torch
import torch.nn as nn
from .blocks import ConvBlock

class Discriminator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.disc1 = ConvBlock(4,64,batch_norm=False)
        self.disc2 = ConvBlock(64,128)
        self.disc3 = ConvBlock(128,256)
        self.disc4 = ConvBlock(256, 512, stride=1) # PatchGAN stride

        self.dic_final = nn.Conv2d(
            in_channels = 512,
            out_channels = 1,
            kernel_size = 4,
            stride= 1,
            padding = 1,    
        )
    

    def forward(self, rgb_tensor, thermal_tensor):

        x = torch.cat((rgb_tensor,thermal_tensor), dim = 1)

        o1 = self.disc1(x)
        o2 = self.disc2(o1)
        o3 = self.disc3(o2)
        o4 = self.disc4(o3)

        o5 = self.dic_final(o4)

        return o5

