import torch
import torch.nn as nn
from src.models.blocks import ConvBlock, DecoderBlock


class Generator(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Encoder
        self.encoder1 = ConvBlock(3, 64, batch_norm=False)
        self.encoder2 = ConvBlock(64, 128)
        self.encoder3 = ConvBlock(128, 256)
        self.encoder4 = ConvBlock(256, 512)
        self.encoder5 = ConvBlock(512, 512)
        self.encoder6 = ConvBlock(512, 512)
        self.encoder7 = ConvBlock(512, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(
            in_channels=512,
            out_channels=512,
            batch_norm=False
        )
        
        # Decoder
        self.decoder1 = DecoderBlock(512, 512)
        self.decoder2 = DecoderBlock(1024, 512)
        self.decoder3 = DecoderBlock(1024, 512)
        self.decoder4 = DecoderBlock(1024, 512)
        self.decoder5 = DecoderBlock(1024, 256)
        self.decoder6 = DecoderBlock(512, 128)
        self.decoder7 = DecoderBlock(256, 64)

        # Output
        self.final = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=128,
                out_channels=1,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.Tanh(),
        )
    
    def forward(self, x):

        # Encoder
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)
        e5 = self.encoder5(e4)
        e6 = self.encoder6(e5)
        e7 = self.encoder7(e6)

        # Bottleneck
        b = self.bottleneck(e7)

        # Decoder
        d1 = self.decoder1(b)
        d2 = self.decoder2(nn.cat([d1, e7], dim=1))
        d3 = self.decoder3(nn.cat([d2, e6], dim=1))
        d4 = self.decoder4(nn.cat([d3, e5], dim=1))
        d5 = self.decoder5(nn.cat([d4, e4], dim=1))
        d6 = self.decoder6(nn.cat([d5, e3], dim=1))
        d7 = self.decoder7(nn.cat([d6, e2], dim=1))

        out = self.final(nn.cat([d7, e1], dim=1))

        return out