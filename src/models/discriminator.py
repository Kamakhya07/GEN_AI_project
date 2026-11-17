import torch
import torch.nn as nn

class PatchGANDiscriminator(nn.Module):
    """
    70x70 PatchGAN Discriminator.
    Evaluates whether overlapping N x N patches of the restored/sharp image
    conditioned on the blurry input are real or fake.
    """
    def __init__(self, in_channels: int = 6, ndf: int = 64):
        super().__init__()

        def disc_block(in_feat: int, out_feat: int, normalize: bool = True, stride: int = 2):
            layers = [nn.Conv2d(in_feat, out_feat, kernel_size=4, stride=stride, padding=1, bias=not normalize)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            *disc_block(in_channels, ndf, normalize=False, stride=2),  # -> 64x64
            *disc_block(ndf, ndf * 2, stride=2),                       # -> 32x32
            *disc_block(ndf * 2, ndf * 4, stride=2),                   # -> 16x16
            *disc_block(ndf * 4, ndf * 8, stride=1),                   # -> 15x15
            nn.Conv2d(ndf * 8, 1, kernel_size=4, stride=1, padding=1)  # -> 14x14 patch map
        )

    def forward(self, blurry: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Concatenate condition (blurry) and target (sharp or fake) along channels
        x = torch.cat([blurry, target], dim=1)
        return self.model(x)
