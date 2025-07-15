import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNet, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Expanding path (Decoder)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256 * 2, 128)
        self.upconv2 = self.upconv_block(128 * 2, 64)
        self.upconv1 = self.upconv_block(64 * 2, 32)

        # Final output layer (1x1 convolution)
        self.final = nn.Conv2d(32 * 2, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """2 consecutive convolution layers with ReLU and Batch Normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        """Up-convolution block with transposed convolution and a conv block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels),
        )

    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expanding path
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat([up4, enc4], 1)  # Skip connection
        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, enc3], 1)  # Skip connection
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, enc2], 1)  # Skip connection
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, enc1], 1)  # Skip connection

        # Final output layer
        return self.final(up1)

class UMask(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UMask, self).__init__()

        # Contracting path (Encoder)
        self.enc1 = self.conv_block(in_channels, 32)
        self.enc2 = self.conv_block(32, 64)
        self.enc3 = self.conv_block(64, 128)
        self.enc4 = self.conv_block(128, 256)

        # Bottleneck
        self.bottleneck = self.conv_block(256, 512)

        # Expanding path (Decoder)
        self.upconv4 = self.upconv_block(512, 256)
        self.upconv3 = self.upconv_block(256 * 2, 128)
        self.upconv2 = self.upconv_block(128 * 2, 64)
        self.upconv1 = self.upconv_block(64 * 2, 32)

        # Final output layer (1x1 convolution)
        self.final = nn.Conv2d(32 * 2, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        """2 consecutive convolution layers with ReLU and Batch Normalization."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            #nn.ReLU(inplace=True),
        )

    def upconv_block(self, in_channels, out_channels):
        """Up-convolution block with transposed convolution and a conv block."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2),
            self.conv_block(out_channels, out_channels),
        )

    def forward(self, x):
        # Contracting path
        enc1 = self.enc1(x)
        enc2 = self.enc2(F.max_pool2d(enc1, 2))
        enc3 = self.enc3(F.max_pool2d(enc2, 2))
        enc4 = self.enc4(F.max_pool2d(enc3, 2))

        # Bottleneck
        bottleneck = self.bottleneck(F.max_pool2d(enc4, 2))

        # Expanding path
        up4 = self.upconv4(bottleneck)
        up4 = torch.cat([up4, enc4], 1)  # Skip connection
        up3 = self.upconv3(up4)
        up3 = torch.cat([up3, enc3], 1)  # Skip connection
        up2 = self.upconv2(up3)
        up2 = torch.cat([up2, enc2], 1)  # Skip connection
        up1 = self.upconv1(up2)
        up1 = torch.cat([up1, enc1], 1)  # Skip connection

        # Final output layer
        return self.final(up1)
