import torch
import torch.nn as nn

class UNetDownBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, normalize: bool = True, dropout: float = 0.0):
        super().__init__()
        layers = [nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not normalize)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)

class UNetUpBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        # Concatenate along channel dimension
        x = torch.cat([x, skip], dim=1)
        return x

class UNetGenerator(nn.Module):
    """
    Compact U-Net Generator for Deblurring Road Patches.
    Takes 3-channel blurry image tensor (range [-1, 1]) and predicts 3-channel restored image tensor.
    """
    def __init__(self, in_channels: int = 3, out_channels: int = 3, ngf: int = 64):
        super().__init__()

        # Encoder (Downsampling)
        # Input: 128x128
        self.down1 = UNetDownBlock(in_channels, ngf, normalize=False)        # -> 64x64
        self.down2 = UNetDownBlock(ngf, ngf * 2)                             # -> 32x32
        self.down3 = UNetDownBlock(ngf * 2, ngf * 4)                         # -> 16x16
        self.down4 = UNetDownBlock(ngf * 4, ngf * 8)                         # -> 8x8
        self.down5 = UNetDownBlock(ngf * 8, ngf * 8)                         # -> 4x4

        # Decoder (Upsampling with skip connections)
        self.up1 = UNetUpBlock(ngf * 8, ngf * 8, dropout=0.2)                # 4x4 -> 8x8, cat -> ngf*16
        self.up2 = UNetUpBlock(ngf * 16, ngf * 4)                           # 8x8 -> 16x16, cat -> ngf*8
        self.up3 = UNetUpBlock(ngf * 8, ngf * 2)                            # 16x16 -> 32x32, cat -> ngf*4
        self.up4 = UNetUpBlock(ngf * 4, ngf)                                # 32x32 -> 64x64, cat -> ngf*2

        # Final head
        self.final = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, out_channels, kernel_size=4, stride=2, padding=1), # 64x64 -> 128x128
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Downward path
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)

        # Upward path with skip connections
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)

        out = self.final(u4)
        return out
