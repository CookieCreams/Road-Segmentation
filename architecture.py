import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        # Upsample using ConvTranspose
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)

        # After concatenation, input channels = (upsampled_channels + skip_channels)
        combined_channels = (in_channels // 2) + skip_channels

        self.conv = nn.Sequential(
            nn.Conv2d(combined_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, skip):
        x = self.up(x)
        # MobileNetV2 usually matches dimensions, but we cat anyway
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class MobileNetV2_UNet(nn.Module):
    def __init__(self, n_class):
        super().__init__()

        # 1. Load Pre-trained Backbone
        backbone = models.mobilenet_v2(weights='IMAGENET1K_V1').features

        # 2. Encoder: Extract specific layers for skip connections
        self.layer0 = backbone[0:2]   # Output: 16 channels, Size: 1/2 (112x112)
        self.layer1 = backbone[2:4]   # Output: 24 channels, Size: 1/4 (56x56)
        self.layer2 = backbone[4:7]   # Output: 32 channels, Size: 1/8 (28x28)
        self.layer3 = backbone[7:14]  # Output: 96 channels, Size: 1/16 (14x14)
        self.layer4 = backbone[14:18] # Output: 320 channels, Bottleneck, Size: 1/32 (7x7)

        # 3. Decoder: Defined to match the asymmetric channels of MobileNetV2
        # DecoderBlock(in_channels, skip_channels, out_channels)
        self.up1 = DecoderBlock(320, 96, 128) # Input 320, Skip 96 -> 128
        self.up2 = DecoderBlock(128, 32, 64)  # Input 128, Skip 32 -> 64
        self.up3 = DecoderBlock(64, 24, 32)   # Input 64, Skip 24 -> 32
        self.up4 = DecoderBlock(32, 16, 16)   # Input 32, Skip 16 -> 16

        # 4. Final Head: Map to number of classes
        # We need one last upsample to get from 112x112 back to original 224x224
        self.out_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)
        self.final_conv = nn.Conv2d(16, n_class, kernel_size=1)

    def forward(self, x):
        # Encoder path
        s0 = self.layer0(x)     # 16 channels
        s1 = self.layer1(s0)    # 24 channels
        s2 = self.layer2(s1)    # 32 channels
        s3 = self.layer3(s2)    # 96 channels
        x = self.layer4(s3)     # 320 channels (Bottleneck)

        # Decoder path with skip connections
        x = self.up1(x, s3)
        x = self.up2(x, s2)
        x = self.up3(x, s1)
        x = self.up4(x, s0)

        # Final upsampling to original input size
        x = self.out_up(x)
        return self.final_conv(x)
