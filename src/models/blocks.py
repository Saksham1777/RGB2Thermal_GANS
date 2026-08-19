import torch.nn as nn

class ConvBlock(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        batch_norm=True,
        negative_slope=0.2,
    ):
        super().__init__()
        layers = [
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias = not batch_norm, # Conv bias is redundant when BatchNorm is active
            )
        ] 

        if batch_norm:
                    batch_layer = nn.BatchNorm2d(out_channels)
                    layers.append(batch_layer)      

        # Activation Layer
        activation_layer = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        layers.append(activation_layer)

        self.block = nn.Sequential(*layers) # "*" for Take every item in this list and pass it as a separate argument.

    def forward(self, x):
        return self.block(x)
    
class DecoderBlock(nn.Module):

    def __init__(self,
        in_channels,
        out_channels,
        kernel_size=4,
        stride=2,
        padding=1,
        batch_norm=True,
        use_dropout = False
        ):
        super().__init__()
        layers = [
             nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=not batch_norm,
            )
        ]

        if batch_norm:
                    batch_layer = nn.BatchNorm2d(out_channels)
                    layers.append(batch_layer)
        
        # Activation Layer
        activation_layer = nn.ReLU(inplace=True)
        layers.append(activation_layer)

        if use_dropout:
            layers.append(nn.Dropout(0.5))

        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)