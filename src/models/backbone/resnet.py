import torch
import torch.nn as nn
from .utils import Module

__all__ = [
    'ResNet', 'ResNet18', 'ResNet34', 'ResNet50',
]


class ResNet(Module):

    def __init__(self, blocks, expansion=1, in_channels=3, num_classes=10):
        super().__init__()
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7,
                      stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.blocks = blocks
        self.avgpool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten()
        )
        self.fc_in = 512 * expansion
        self.fc = nn.Sequential(
            nn.Linear(512 * expansion, num_classes)
        )
        self.initialize_weights()

    def forward(self, x):
        x = self.head(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        x = self.fc(x)
        return x


class ResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, stride=stride,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.connection = None
        if in_channels != out_channels or stride != 1:
            self.connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x) + identity
        return self.activation(x)


class BNResBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride,
                      kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4,
                      kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4)
        )
        self.connection = None
        if in_channels != out_channels * 4 or stride != 1:
            self.connection = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4,
                          kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4)
            )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.connection is not None:
            identity = self.connection(x)
        x = self.block(x)
        x += identity
        return self.activation(x)


class ResNet18(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        blocks = []
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))

        blocks.append(ResBlock(64, 128, stride=2))
        blocks.append(ResBlock(128, 128))

        blocks.append(ResBlock(128, 256, stride=2))
        blocks.append(ResBlock(256, 256))

        blocks.append(ResBlock(256, 512, stride=2))
        blocks.append(ResBlock(512, 512))

        blocks = nn.Sequential(*blocks)
        super().__init__(blocks, 1, in_channels, num_classes)


class ResNet34(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        blocks = []
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))
        blocks.append(ResBlock(64, 64))

        blocks.append(ResBlock(64, 128, stride=2))
        blocks.append(ResBlock(128, 128))
        blocks.append(ResBlock(128, 128))
        blocks.append(ResBlock(128, 128))

        blocks.append(ResBlock(128, 256, stride=2))
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 256))
        blocks.append(ResBlock(256, 256))

        blocks.append(ResBlock(256, 512, stride=2))
        blocks.append(ResBlock(512, 512))
        blocks.append(ResBlock(512, 512))

        blocks = nn.Sequential(*blocks)
        super().__init__(blocks, 1, in_channels, num_classes)


class ResNet50(ResNet):
    def __init__(self, in_channels=3, num_classes=10):
        blocks = []
        blocks.append(BNResBlock(64, 64))
        blocks.append(BNResBlock(64 * 4, 64))
        blocks.append(BNResBlock(64 * 4, 64))

        blocks.append(BNResBlock(64 * 4, 128, stride=2))
        blocks.append(BNResBlock(128 * 4, 128))
        blocks.append(BNResBlock(128 * 4, 128))
        blocks.append(BNResBlock(128 * 4, 128))

        blocks.append(BNResBlock(128 * 4, 256, stride=2))
        blocks.append(BNResBlock(256 * 4, 256))
        blocks.append(BNResBlock(256 * 4, 256))
        blocks.append(BNResBlock(256 * 4, 256))

        blocks.append(BNResBlock(256 * 4, 512, stride=2))
        blocks.append(BNResBlock(512 * 4, 512))
        blocks.append(BNResBlock(512 * 4, 512))

        blocks = nn.Sequential(*blocks)
        super().__init__(blocks, 4, in_channels, num_classes)
