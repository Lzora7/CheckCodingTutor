import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_classes=1):
        super(SimpleUNet, self).__init__()  # наследуемся от nn.Module

        # ----- ЭНКОДЕР (2 уровня) -----
        # Encoder level 1: conv -> conv -> pool
        self.enc1_conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.enc1_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder level 2
        self.enc2_conv1 = nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1)
        self.enc2_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Боттлнек
        self.bottleneck_conv1 = nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1)

        # ----- ДЕКОДЕР (2 уровня) -----
        # Decoder level 1: upconv -> conv -> conv
        self.up1 = nn.ConvTranspose2d(base_channels * 4, base_channels * 2, kernel_size=2, stride=2)
        self.dec1_conv1 = nn.Conv2d(base_channels * 2 + base_channels * 2, base_channels * 2, kernel_size=3, padding=1)
        self.dec1_conv2 = nn.Conv2d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1)

        # Decoder level 2
        self.up2 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec2_conv1 = nn.Conv2d(base_channels + base_channels, base_channels, kernel_size=3, padding=1)
        self.dec2_conv2 = nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1)

        # Финальный слой
        self.out_conv = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, x):
        # ----- Encoder -----
        # level 1
        x1 = F.relu(self.enc1_conv1(x))
        x1 = F.relu(self.enc1_conv2(x1))
        p1 = self.pool1(x1)

        # level 2
        x2 = F.relu(self.enc2_conv1(p1))
        x2 = F.relu(self.enc2_conv2(x2))
        p2 = self.pool2(x2)

        # bottleneck
        b = F.relu(self.bottleneck_conv1(p2))
        b = F.relu(self.bottleneck_conv2(b))

        # ----- Decoder -----
        # level 1
        # u1 = self.up1(b)                 # upsample
        u1 = torch.cat([u1, x2], dim=1)  # skip-connection
        u1 = F.relu(self.dec1_conv1(u1))
        u1 = F.relu(self.dec1_conv2(u1))

        # level 2
        u2 = self.up2(u1)
        u2 = torch.cat([u2, x1], dim=1)
        u2 = F.relu(self.dec2_conv1(u2))
        u2 = F.relu(self.dec2_conv2(u2))

        out = self.out_conv(u2)
        return out


# Простой тест на корректность форм
if __name__ == "__main__":
    model = SimpleUNet(in_channels=3, base_channels=16, num_classes=1)
    x = torch.randn(2, 3, 128, 128)  # [B, C, H, W]
    y = model(x)
    print("Input shape :", x.shape)
    print("Output shape:", y.shape)  # должно проходить без ошибок формы
